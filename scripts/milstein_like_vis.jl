using DrWatson
@quickactivate "MirrorCellNetwork"

using SpikingNeuralNetworks
SNN.@load_units

using CairoMakie

include(srcdir("btsp_functions.jl"))

# milstein like place cell network (Madar et al. 2025 Fig3/S13A)
# params from Table S1

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
SNN.monitor!(E, :v,sr=100Hz,monitor_time=600*s)
SNN.monitor!(syn_dummy, [:pre_trace,:post_trace]; sr = 20Hz,variables=:LTPVars,monitor_time=600*s)

model = SNN.compose(; E, Ids, stims,syn,CS_trigger,Id_post,syn_dummy,norm_dummy,silent=true)
SNN.sim!(model, duration = 60*s,dt=10*ms)
SNN.train!(model, duration = 30*s,dt=10*ms)

#over write real IF cell's weights using what the dummy neuron learned
syn.W = syn_dummy.W
SNN.sim!(model, duration = 70*s,dt=10*ms)

interval = 0:500ms:160*s 
fr, r = SNN.bin_spiketimes(E;interval)
fr = fr*2

spike_pos_1 = mod.((SNN.spiketimes(E)[1])*vs[1][:vel],vs[1][:L])
spike_lap_1 = floor.(SNN.spiketimes(E)[1]/20s)

spike_pos_2 = mod.((SNN.spiketimes(E)[2])*vs[1][:vel],vs[1][:L])
spike_lap_2 = floor.(SNN.spiketimes(E)[2]/20s)

fig = Figure()
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[2,1])
ax3 = Axis(fig[3,1])
ax_scatter = Axis(fig[:,2],limits=(0,300,nothing,nothing),yreversed = true)

SNN.vecplot!(ax1,E,:v,add_spikes=true)
lines!(ax2,r,vec(fr[1,:]))
lines!(ax2,r,vec(fr[2,:]))
SNN.vecplot!(ax3,syn_dummy,:LTPVars_post_trace)
linkxaxes!(ax1, ax2,ax3)
hidexdecorations!(ax1, grid = false)
hidexdecorations!(ax2, grid = false)

scatter!(ax_scatter,spike_pos_1,spike_lap_1,markersize=4)
scatter!(ax_scatter,spike_pos_2,spike_lap_2,markersize=4)
fig
