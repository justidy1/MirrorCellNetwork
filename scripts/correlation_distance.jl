using DrWatson
@quickactivate "MirrorCellNetwork"

using SpikingNeuralNetworks
SNN.@load_units

using CairoMakie
using DSP

include(srcdir("btsp_functions.jl"))

function multiple_CS(t_stims,w_init,A_BTSP; 
    N_CA3 = 600,sim_duration=80*s,dt=10*ms,vel = 15*(cm/s), L_tl = 600, CA3_pf_peak_rate = 10*Hz,CA3_pf_σ = 18*cm)
    # rate of place field centered at x0
    function gaussian_pf(x,x0,peak_f=CA3_pf_peak_rate,σ=CA3_pf_σ)
        return peak_f*exp(-(x-x0)^2 / (2*σ^2))
    end
    
    Ids_pre = SNN.Identity(N=N_CA3)
    Ids_post = SNN.Identity(N=1)
    
    total_track_len = L_tl*cm
    x0s = LinRange(0,L_tl,Ids_pre.N)*cm

    vs =  [Dict{Symbol,Any}(:vel => vel,:x0 => x0,:L => total_track_len) for x0 in x0s]
    pf_r(t,v) = gaussian_pf(mod(t*v[:vel],v[:L]),v[:x0])

    pf_params = [SNN.PoissonVariable(variables = v,rate=pf_r) for v in vs]
    stims = [SNN.Stimulus(param, Ids_pre, :ge, neurons = [i]) for (i,param) in enumerate(pf_params)]

    param = SNN.SpikeTimeParameter(t_stims, ones(Int64,length(t_stims)))
    CS_trigger = SNN.SpikeTimeStimulusIdentity(Ids_post, :g; param)
    
    W_init = fill(w_init,1,Ids_pre.N) # flat initial weights
    syn_dummy = SNN.SNNModels.SpikingSynapse(Ids_pre, Ids_post, :g; 
            conn = W_init, 
            LTPParam = SNN.SNNModels.BTSPMadar(p_CS=1,
                        W_max=Inf*pA,
                        A_BTSP_pre = A_BTSP,
                        A_BTSP_post = A_BTSP))

    syn_dummy.g = zeros(Float32, Ids_post.N) # transmit into a dummy buffer: post fires only from its stimulus above
    norm_dummy = SNN.SynapseNormalization([syn_dummy]; param = SNN.MultiplicativeNorm(τ = 1s))

    model = SNN.compose(; Ids_pre, Ids_post, stims,CS_trigger,syn_dummy,norm_dummy,silent=true)
    SNN.train!(model, duration = sim_duration,dt=dt)
    W = syn_dummy.W

    gauss_win = gaussian(length(W),0.01)
    gauss_win = gauss_win/sum(gauss_win)
    lap_time = (L_tl*cm/vel)
    W_smooth = conv(W,gauss_win)[Int(length(W)/2):end-Int(length(W)/2)]
    intended_CS_locs = round.(Int,length(W).*(t_stims/lap_time))

    return W,W_smooth,W_smooth[intended_CS_locs],intended_CS_locs
end

function multiple_CS_trained_IF(W_init,cs_times; 
    N_CA3 = 600,sim_duration=40*s,dt=10*ms,vel = 15*(cm/s), L_tl = 600, CA3_pf_peak_rate = 10*Hz,CA3_pf_σ = 18*cm)
    # rate of place field centered at x0
    function gaussian_pf(x,x0,peak_f=CA3_pf_peak_rate,σ=CA3_pf_σ)
        return peak_f*exp(-(x-x0)^2 / (2*σ^2))
    end
    
    Ids_pre = SNN.Identity(N=N_CA3)
    Ids_post = SNN.IF(N=1,param = SNN.IFParameter(R = 100*MΩ,
                τm=20*ms,Vt=-54*mV),
            synapse = SNN.DeltaSynapse())
    
    total_track_len = L_tl*cm
    x0s = LinRange(0,L_tl,Ids_pre.N)*cm

    vs =  [Dict{Symbol,Any}(:vel => vel,:x0 => x0,:L => total_track_len) for x0 in x0s]
    pf_r(t,v) = gaussian_pf(mod(t*v[:vel],v[:L]),v[:x0])

    pf_params = [SNN.PoissonVariable(variables = v,rate=pf_r) for v in vs]
    stims = [SNN.Stimulus(param, Ids_pre, :ge, neurons = [i]) for (i,param) in enumerate(pf_params)]
    
    syn = SNN.SNNModels.SpikingSynapse(Ids_pre, Ids_post, :ge; 
            conn = reshape(W_init*pA,Ids_post.N,Ids_pre.N))
    
    norm = SNN.SynapseNormalization([syn]; param = SNN.MultiplicativeNorm(τ = 1s))

    SNN.monitor!([Ids_post,Ids_pre],:fire,sr=100Hz,monitor_rate = 1000Hz,
                monitor_time=sim_duration)
    
    model = SNN.compose(; Ids_pre, Ids_post,stims,syn,norm,silent=true)
    SNN.sim!(model, duration = sim_duration,dt=dt)
    
    st = SNN.spiketimes(Ids_post)
    interval = 0:500*ms:sim_duration
    fr, _ = SNN.firing_rate(st; neurons = 1,interval = interval, τ = 500ms)
    
    return fr,fr.(1,cs_times)
end

params = Dict(
    "W_init" => collect(1:1:50),
    "A_BTSP" => collect(5:1:50),
    "ts_CS" => [[10*s,30*s]]
)
dicts = dict_list(params)

function make_sim(d::Dict)
    @unpack W_init,A_BTSP,ts_CS = d
    W,W_smooth,W_peaks,_ = multiple_CS(ts_CS,W_init*pA,A_BTSP*pA)
    fr,fr_peaks = multiple_CS_trained_IF(W_smooth,ts_CS)

    fulld = copy(d)
    fulld["W_peaks"] = W_peaks
    fulld["W"] = W
    fulld["W_smooth"] = W_smooth
    fulld["fr"] = fr
    fulld["fr_peaks"] = fr_peaks

    return fulld
end

for d in dicts
    produce_or_load(make_sim,d,datadir("sims","ts_CS_IF_simulations"),filename=savename(d))
end

using DataFrames

df = coalesce.(collect_results(datadir("sims","ts_CS_IF_simulations")))

w1 = [w[1] for w in df.W_peaks]
w2 = [w[2] for w in df.W_peaks]
fr1 = [fr[1] for fr in df.fr_peaks]
fr2 = [fr[2] for fr in df.fr_peaks]

fig = Figure()
ax1 = Axis(fig[1,1],aspect=1)
ax2 = Axis(fig[1,2],aspect=1)
ax3 = Axis(fig[1,4],aspect=1)
w_colorrange = (minimum(vcat(df.W_peaks...)),maximum(vcat(df.W_peaks...)))
hm_w1 = heatmap!(ax1,df.W_init,df.A_BTSP, w1;colorrange = w_colorrange)
hm_w2 = heatmap!(ax2,df.W_init,df.A_BTSP, w2;colorrange =w_colorrange)

Colorbar(fig[1, 3],
        colorrange = w_colorrange)

hm_DI = heatmap!(ax3,df.W_init,df.A_BTSP, (w2.-w1)./(w2.+w1))
Colorbar(fig[1,5], hm_DI)

linkyaxes!(ax1, ax2)
linkyaxes!(ax2, ax3)
hideydecorations!(ax2; grid=false)
hideydecorations!(ax3; grid=false)
rowsize!(fig.layout, 1, Aspect(1,1.0))

ax21 = Axis(fig[2,1],aspect=1)
ax22 = Axis(fig[2,2],aspect=1)
ax23 = Axis(fig[2,4],aspect=1)

fr_colorrange = (minimum(vcat(df.fr_peaks...)),maximum(vcat(df.fr_peaks...)))
hm_fr1 = heatmap!(ax21,df.W_init,df.A_BTSP, fr1)
hm_fr2 = heatmap!(ax22,df.W_init,df.A_BTSP, fr2)

Colorbar(fig[2, 3],
        colorrange = fr_colorrange)

hm_fr_DI = heatmap!(ax23,df.W_init,df.A_BTSP, abs.(fr2.-fr1)./(fr2.+fr1))
Colorbar(fig[2,5], hm_fr_DI)

linkyaxes!(ax21, ax22)
linkyaxes!(ax22, ax23)
hideydecorations!(ax22; grid=false)
hideydecorations!(ax23; grid=false)
resize_to_layout!(fig)
rowsize!(fig.layout, 2, Aspect(1,1.0))
fig

# W,W_smooth,W_peaks,locs = multiple_CS([10*s,30*s],20*pA,20*pA)

# fig = Figure()
# ax1 = Axis(fig[1,1])
# lines!(ax1,W)
# lines!(ax1,W_smooth)
# scatter!(ax1,locs,W_peaks,color=:black,markersize=10)
# fig
