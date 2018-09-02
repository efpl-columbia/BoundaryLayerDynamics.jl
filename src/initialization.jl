# nodes generally start at zero, vertically direction is centered for uvp-nodes
@inline coord(i, δ, ::Val{:uvp}) = (δ[1] * (i[1]-1),
                                    δ[2] * (i[2]-1),
                                    δ[3] * (2*i[3]-1)/2)
@inline coord(i, δ, ::Val{:w})   = (δ[1] * (i[1]-1),
                                    δ[2] * (i[2]-1),
                                    δ[3] * (i[3]-1))

initialize!(cf::ChannelFlowProblem, u::Tuple) = initialize!(cf, u...)
initialize!(cf::ChannelFlowProblem{T}, u0) where T =
    initialize!(cf, u0, (x,y,z) -> zero(T))
initialize!(cf::ChannelFlowProblem{T}, u0, v0) where T =
    initialize!(cf, u0, v0, (x,y,z) -> zero(T))

function initialize!(cf::ChannelFlowProblem, u0, v0, w0)
    δ_big = cf.grid.l ./ cf.tf_big.n
    for (vel_hat, vel_0, nodes) in zip(cf.vel_hat, (u0, v0, w0), (Val(:uvp), Val(:uvp), Val(:w)))
        initialize!(vel_hat, vel_0, δ_big, nodes, cf.tf_big.plan_fwd,
                cf.tf_big.buffers_pd[1], cf.tf_big.buffers_fd[1])
    end
    apply_bcs!(cf)
end

function initialize!(vel_hat, vel_0, δ_big, nodes, plan_big_fwd, buffer_big_pd, buffer_big_fd)
    for i in CartesianIndices(buffer_big_pd)
        buffer_big_pd[i] = vel_0(coord(i, δ_big, nodes)...)
    end
    fft_dealiased!(vel_hat, buffer_big_pd, plan_big_fwd, buffer_big_fd)
end
