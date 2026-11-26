using DrWatson
@quickactivate "MirrorCellNetwork"

using SpikingNeuralNetworks
SNN.@load_units

using CairoMakie

include(srcdir("btsp_functions.jl"))

###############
# SET UP
###############
Ids_pre = SNN.Identity(N=200)
Ids_post = SNN.Identity(N=1)

# rate of place field centered at x0
function gaussian_pf(x,x0,peak_f=10*Hz,σ=18*cm)
    return peak_f*exp(-(x-x0)^2 / (2*σ^2))
end

vel = 15*(cm/s)
L_tl = 600
total_track_len = L_tl*cm
x0s = LinRange(0,L_tl,Ids_pre.N)*cm

vs =  [Dict{Symbol,Any}(:vel => vel,:x0 => x0,:L => total_track_len) for x0 in x0s]

pf_r(t,v) = gaussian_pf(mod(t*v[:vel],v[:L]),v[:x0])

pf_params = [SNN.PoissonVariable(variables = v,rate=pf_r) for v in vs]
stims = [SNN.Stimulus(param, Ids_pre, :ge, neurons = [i]) for (i,param) in enumerate(pf_params)]

# CS triggers
param = SNN.SpikeTimeParameter([10*s,30*s], [1,1])
CS_trigger = SNN.SpikeTimeStimulusIdentity(Ids_post, :g; param)

W_init = fill(20*pA,1,Ids_pre.N) # flat initial weights
syn_dummy = SNN.SNNModels.SpikingSynapse(Ids_pre, Ids_post, :g; 
        conn = W_init, 
        LTPParam = SNN.SNNModels.BTSPMadar(p_CS=1,
                    W_max=Inf*pA,
                    A_BTSP_pre = 20*pA,
                    A_BTSP_post = 20*pA))
syn_dummy.g = zeros(Float32, Ids_post.N) # transmit into a dummy buffer: post fires only from its stimulus above
norm_dummy = SNN.SynapseNormalization([syn_dummy]; param = SNN.MultiplicativeNorm(τ = 1s))

SNN.monitor!([Ids_pre,Ids_post,CS_trigger], :fire, sr=100Hz, monitor_time=600*s)

model = SNN.compose(; Ids_pre, Ids_post, stims,CS_trigger,syn_dummy,norm_dummy,silent=true)

SNN.train!(model, duration = 80*s,dt=10*ms)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,syn_dummy.W)
fig