using SpikingNeuralNetworks
SNN.@load_units

@eval SNN.SNNModels begin

    abstract type BTSPParameter <: LTPParameter end
    ###
    #Madar, Antoine D., et al. 
    #"Synaptic plasticity rules driving representational shifting in the hippocampus." 
    #Nature neuroscience 28.4 (2025): 848-860.
    ###
    @snn_kw struct BTSPMadar{FT = Float32} <: BTSPParameter
        A_BTSP_pre::FT = 20.0*pA
        A_BTSP_post::FT = 20.0*pA 
        b::FT = 1.1 # b in Eq. 18 of Madar et al. 2025
        τpre_before_post::FT = 1.31*s
        τpost_before_pre::FT = 0.69*s
        p_CS::FT = 0.01 # the probability a presynaptic spike is converted to a complex spike (CS) instructive signal
        W_max::FT = 80.0*pA
        W_min::FT = 0.0*pA
    end

    @snn_kw struct BTSPVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
        Npost::IT # number of post-synaptic neurons
        Npre::IT # number of pre-synaptic neurons
        pre_trace::VFT = zeros(Float32,Npre) # trace for pre-synaptic signal
        post_trace::VFT = zeros(Float32,Npost) # trace for post-synaptic signal
        active::VBT = [true]
    end

    function plasticityvariables(param::T,Npre,Npost) where {T<:BTSPParameter}
        return BTSPVariables(Npre = Npre, Npost = Npost)
    end

    function plasticity!(c::PT, 
        param::BTSPMadar, 
        vars::BTSPVariables, 
        dt::AbstractFloat, 
        T::Time) where {PT<:AbstractSparseSynapse}
        
        @unpack A_BTSP_pre,A_BTSP_post,b,τpre_before_post,τpost_before_pre,p_CS,W_max,W_min = param
        @unpack Npost,Npre,pre_trace,post_trace = vars

        t = get_time(T)

        # put weights in bounds
        c.W = clamp.(c.W,W_min,W_max)

        # get index of all pre and post neurons which emitted spikes in this time step
        pre_spikes,post_spikes = presynaptic_spikes(c),postsynaptic_spikes(c)
        
        # subsample the post-synaptic spikes
        CS_spikes = post_spikes[rand(length(post_spikes)).<p_CS]
        
        # FIRST use traces to update weights incoming to post-synaptic neurons that experience CS
        # based on spikes that occur before the CS
        @inbounds for i = CS_spikes
            input_neurons = presynaptic(c,Int64(i))
            for j = input_neurons
                s = indices(c,[j],[i])
                c.W[s] = clamp.(c.W[s] .+ A_BTSP_pre*pre_trace[j],W_min,W_max)
            end
        end

        # THEN for all pre-synaptic spikes apply their weight updates based on post-CS trace
        @inbounds for i = pre_spikes
            s = postsynaptic_idxs(c,Int64(i))
            c.W[s] = clamp.(c.W[s] .+ A_BTSP_post*b*post_trace[postsynaptic(c,Int64(i))],W_min,W_max)
        end

        # LAST, update traces (decay, then add +1 for spikes/CS)

        # update pre_trace using pre_synaptic spikes
        @inbounds for i = pre_spikes
            pre_trace[i] += 1
        end
        # same for post trace
        @inbounds for i = CS_spikes
            post_trace[i] += 1
        end

        @inbounds for i = 1:Npre
            pre_trace[i] *= exp(-dt/τpre_before_post)
        end

        @inbounds for i = 1:Npost
            post_trace[i] *= exp(-dt/τpost_before_pre)
        end

        return nothing
    end
end

# a function which computed LTP for pre-CS pair of ΔT separation
function btsp_test(btsp_param; ΔT)
    pre = Identity(N = 1, name = "pre")
    post = Identity(N = 1, name = "post")
    stim_pre = SNN.SNNModels.SpikeTimeStimulusIdentity(pre, :g, param = SpikeTimeParameter([5*s + ΔT], [1]))
    stim_post =  SNN.SNNModels.SpikeTimeStimulusIdentity(post, :g, param = SpikeTimeParameter([5.001*s], [1]))
    w = ones(Float32, 1, 1)
    syn = SNN.SNNModels.SpikingSynapse(pre, post, :g, conn = w, LTPParam = btsp_param)
    syn.g = zeros(Float32, post.N) # transmit into a dummy buffer: post fires only from its stimulus
    model = compose(; pre, post, stim_pre, stim_post, syn, silent = true)
    train!(model = model, duration = 10*s,dt=10*ms)

    return syn.W[1]-1
end