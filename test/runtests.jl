using DrWatson, Test, StatsBase, Random
@quickactivate "MirrorCellNetwork"

# Here you include files using `srcdir`
include(srcdir("btsp_functions.jl"))

# BTSP testing funcitons
function BTSP_values_against_GT()
    ΔTs = [-4,-2,0,2,4]*s
    ΔWs = zeros(length(ΔTs))

    for (i,ΔT) in enumerate(ΔTs)
        ΔWs[i]=btsp_test(SNN.SNNModels.BTSPMadar(p_CS=1,W_max=2000*pA);ΔT)
    end

    # ground truth Madar Fig. S13 panel A, normalized to max
    # extracted from webplotdigitizer
    Δt_gt = [-4,-2,0,2,4]
    w_max = [0.0468,0.216,1,0.0534,0]
    W_sim = [ΔWs[(ΔTs/1000).==t][1] for t in Δt_gt]/maximum(ΔWs)

    return sqrt(sum((W_sim-w_max).^2)/length(w_max)) < 0.01
end

# Give 2 cells 2 different CS after a few laps, do their place fields move like expected?
function Milstein_test()
    # milstein like place cell network (Madar et al. 2025 Fig3/S13A)
    # params from Table S1, expect Delta Synapse instead of exp synapse
    # and p_CS = 1 since we are triggering it manually at pre-specified times
    # using Identity() cells
    # CS are triggered where the cell was silent initally
    # also, cells recieved identical place-modulated spike inputs
    Random.seed!(123)

    # rate of place field centered at x0
    function gaussian_pf(x,x0,peak_f=10*Hz,σ=18*cm)
        return peak_f*exp(-(x-x0)^2 / (2*σ^2))
    end

    vel = 15*(cm/s)
    total_track_len = 300*cm
    x0s = LinRange(0,300,100)*cm

    vs =  [Dict{Symbol,Any}(:vel => vel,:x0 => x0,:L => total_track_len) for x0 in x0s]

    pf_r(t,v) = gaussian_pf(mod(t*v[:vel],v[:L]),v[:x0])

    E = SNN.IF(N = 2,param = SNN.IFParameter(R = 100*MΩ,
                τm=20*ms,Vt=-54*mV),
            synapse = SNN.DeltaSynapse())
    Ids = SNN.Identity(N=100)
    Id_post = SNN.Identity(N=2)

    pf_params = [SNN.PoissonVariable(variables = v,rate=pf_r) for v in vs]
    stims = [SNN.Stimulus(param, Ids, :ge, neurons = [i]) for (i,param) in enumerate(pf_params)]

    W_init = repeat(reshape(85.0.*exp.(-(vec(1:100).-50).^2 ./ (2*10^2)),1,100),2)
    syn = SNN.SNNModels.SpikingSynapse(Ids, E, :ge; conn = W_init*pA)

    # spike time to trigger CS (p_CS = 1 only during window around this spike)
    param = SNN.SpikeTimeParameter([65*s,75*s], [1,2])
    CS_trigger = SNN.SpikeTimeStimulusIdentity(Id_post, :g; param)

    syn_dummy = SNN.SNNModels.SpikingSynapse(Ids, Id_post, :g, conn = W_init*pA, LTPParam = SNN.SNNModels.BTSPMadar(p_CS=1,W_max=1000*pA))
    syn_dummy.g = zeros(Float32, Ids.N) # transmit into a dummy buffer: post fires only from its stimulus above
    norm_dummy = SNN.SynapseNormalization([syn_dummy]; param = SNN.MultiplicativeNorm(τ = 1s))

    SNN.monitor!([E,Ids,CS_trigger], :fire,sr=100Hz,monitor_time=600*s)

    model = SNN.compose(; E, Ids, stims,syn,CS_trigger,Id_post,syn_dummy,norm_dummy,silent=true)
    SNN.sim!(model, duration = 60*s,dt=10*ms)
    SNN.train!(model, duration = 30*s,dt=10*ms)

    #over write real IF cell's weights using what the dummy neuron learned
    syn.W = syn_dummy.W
    SNN.sim!(model, duration = 70*s,dt=10*ms)

    spike_pos_1 = mod.((SNN.spiketimes(E)[1])*vs[1][:vel],vs[1][:L])
    spike_lap_1 = floor.(SNN.spiketimes(E)[1]/20s)

    spike_pos_2 = mod.((SNN.spiketimes(E)[2])*vs[1][:vel],vs[1][:L])
    spike_lap_2 = floor.(SNN.spiketimes(E)[2]/20s)

    # check where place fields are for neurons
    COM_pre_CS1 = mean(spike_pos_1[spike_lap_1.<4])
    COM_pre_CS2 = mean(spike_pos_2[spike_lap_2.<4])

    COM_post_CS1 = mean(spike_pos_1[spike_lap_1.>5])
    COM_post_CS2 = mean(spike_pos_2[spike_lap_2.>5])

    # are fields at 150 cm initially?, 10 cm of leeway
    pre_check_CS1 = abs(COM_pre_CS1 - 150*cm) < 10*cm
    pre_check_CS2 = abs(COM_pre_CS2 - 150*cm) < 10*cm
    pre_check = pre_check_CS1 & pre_check_CS2

    # are the fields translated to where they should be?
    # CS1 is +5 second new lap, CS2 is +15 new lap
    post_check_CS1 = abs(COM_post_CS1 - (300*(5/20)*cm)) < 10*cm
    post_check_CS2 = abs(COM_post_CS2 - (300*(15/20)*cm)) < 10*cm
    post_check = post_check_CS1 & post_check_CS2

    # auxillary check that firing rates of cells are plausible
    interval = 0:500ms:160*s 
    fr, r = SNN.bin_spiketimes(E;interval)
    fr = fr*2
    max_pre_CS_1 = (maximum(vec(fr[1,r.<90*s])) > 5) & (maximum(vec(fr[1,r.<90*s])) < 100)
    max_pre_CS_2 = (maximum(vec(fr[2,r.<90*s])) > 5) & (maximum(vec(fr[2,r.<90*s])) < 100)

    max_post_CS_1 = (maximum(vec(fr[1,r.>90*s])) > 5) & (maximum(vec(fr[1,r.>90*s])) < 100)
    max_post_CS_2 = (maximum(vec(fr[2,r.>90*s])) > 5) & (maximum(vec(fr[2,r.>90*s])) < 100)

    fr_check = max_pre_CS_1 & max_pre_CS_2 & max_post_CS_1 & max_post_CS_2
    
    return pre_check & post_check & fr_check
end


function test_locality()
    # check private/locality of rule
    # Initally, all weights 85*pA. Id_pre[1] fires in window between 1-2 s
    # Id_pre[2] fires in window between 11-12 s
    # Id_post[1] fires at 1.5 s, Id_post[2] fires at 11.5 s
    # if local, then ONLY Id_pre[1]->Id_post[1] and Id_pre[2]->Id_post[2] will LTP
    # assures that connections change like we expect (ie rule out non-local interactions)
    Random.seed!(123)
    # pop 1 goes off, then pop 2 10 s seconds later
    param1 = SNN.PoissonInterval(rate = 5Hz, intervals = [[1s, 2s]])
    param2 = SNN.PoissonInterval(rate = 5Hz, intervals = [[11s, 12s]])
    Id_pre = SNN.Identity(N=2)

    # neuron 1 fires CS in pop 1 window, neuron 2 in pop 2
    Id_post = SNN.Identity(N=2)

    stim1 = SNN.Stimulus(param1, Id_pre, :ge,neurons=[1])
    stim2 = SNN.Stimulus(param2, Id_pre, :ge,neurons=[2])

    # spike time to trigger CS (p_CS = 1 only during window around this spike)
    param = SNN.SpikeTimeParameter([1.5*s,11.5*s], [1,2])
    CS_trigger = SNN.SpikeTimeStimulusIdentity(Id_post, :g; param)

    syn_dummy = SNN.SNNModels.SpikingSynapse(Id_pre, Id_post, :g, conn = (p = 1, μ=85*pA), 
                LTPParam = SNN.SNNModels.BTSPMadar(p_CS=1,W_max=1000*pA))
    syn_dummy.g = zeros(Float32, Ids.N) # transmit into a dummy buffer: post fires only from its stimulus above

    model = SNN.compose(;Id_pre,stim1,stim2,syn_dummy,CS_trigger,Id_post,silent=true)

    SNN.monitor!([Id_pre,Id_post], :fire,sr=100Hz,monitor_time=600*s)
    SNN.train!(model, duration = 20*s,dt=10*ms)

    is_LTP_1_to_1 = (syn_dummy.W[SNN.indices(syn_dummy, [1], [1])][1] - 85*pA) > 2*pA # 2 pA wiggle room
    is_LTP_2_to_2 = (syn_dummy.W[SNN.indices(syn_dummy, [2], [2])][1] - 85*pA) > 2*pA # 2 pA wiggle room

    is_SAME_1_to_2 = abs(syn_dummy.W[SNN.indices(syn_dummy, [1], [2])][1] - 85*pA) <= 2*pA # 2 pA wiggle room
    is_SAME_2_to_1 = abs(syn_dummy.W[SNN.indices(syn_dummy, [2], [1])][1] - 85*pA) <= 2*pA # 2 pA wiggle room

    return is_LTP_1_to_1 & is_LTP_2_to_2 & is_SAME_1_to_2 & is_SAME_2_to_1
end

# Run test suite
println("Starting tests")
ti = time()


@testset "BTSP Kernel tests" begin
    @test BTSP_values_against_GT()
    @test Milstein_test()
    @test test_locality()
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
