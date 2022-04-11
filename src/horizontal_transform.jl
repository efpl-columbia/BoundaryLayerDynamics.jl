module Transform

using FFTW: FFTW
using LinearAlgebra: mul!
using ..Helpers
using ..Grid: NodeSet, vrange
using ..Domain: SemiperiodicDomain, x1range, x2range, x3range

struct HorizontalTransform{T,N,P1,P2,S,R}
    pdbuffer::Array{T,2}
    fdbuffer::Array{Complex{T},2}
    fwdplan::P1
    bwdplan::P2
    state::S
    rates::R
    # TODO: split fields into state & rates

    function HorizontalTransform(::Type{T}, grid, size, state = [], rates = []; flags=FFTW.MEASURE) where T
        fdsize = (div(size[1],2)+1, size[2])
        fdbuffer = zeros(Complex{T}, fdsize)
        pdbuffer = zeros(T, size)
        fwdplan = FFTW.plan_rfft(pdbuffer, flags=flags)
        bwdplan = FFTW.plan_brfft(fdbuffer, size[1], flags=flags)
        n3(::NodeSet{:C}) = grid.n3c
        n3(::NodeSet{:I}) = grid.n3i
        n3(field) = n3(nodes(Val(field)))
        state = NamedTuple(f => zeros(T, size..., n3(f)) for f in state)
        rates = NamedTuple(f => zeros(T, size..., n3(f)) for f in rates)
        new{T,size,typeof(fwdplan),typeof(bwdplan),typeof(state),typeof(rates)}(pdbuffer, fdbuffer, fwdplan, bwdplan, state, rates)
    end
end

function init_transforms(::Type{T}, grid, processes) where T
    sizes = unique([pdsize(grid)]) # always include default size
    Dict(s => HorizontalTransform(T, grid, s) for s in sizes)
end


# 2D size of physical domain fields
default_size(grid) = pdsize(grid)
pdsize(grid) = pdsize(grid, nothing)
pdsize(grid, dealiasing::Nothing) = (2 + 2 * grid.k1max, 2 + 2 * grid.k2max)
pdsize(grid, dealiasing::Tuple{Int,Int}) = dealiasing
pdsize(grid, dealiasing::Symbol) =
    if dealiasing == :quadratic
        # TODO: better heuristics for optimal size
        (1 + 3 * grid.k1max, 1 + 3 * grid.k2max)
    else
        error("Unsupported dealiasing type: $(dealiasing)")
    end

# 3D size of physical domain fields
pdsize(grid, ::NodeSet{:C}, dealiasing) = (pdsize(grid, dealiasing)..., grid.n3c)
pdsize(grid, ::NodeSet{:I}, dealiasing) = (pdsize(grid, dealiasing)..., grid.n3i)
pdsize(grid, nodes::NodeSet) = pdsize(grid, nodes, nothing)

# returns a range of rational ζ-values between 0 and 1
h1range(grid, dealiasing) = LinRange(0//1, 1//1, pdsize(grid, dealiasing)[1]+1)[1:end-1]
h2range(grid, dealiasing) = LinRange(0//1, 1//1, pdsize(grid, dealiasing)[2]+1)[1:end-1]

function transform_state!(transform, state)
    for (field, values) in pairs(transform.state)
        get_field!(values, transform, state[field])
    end
    transform.state
end

function add_rates!(rates, transform)
    for (field, values) in transform.rates
        println("adding rate: ", field)
        add_layer(rates[field], transform, values)
    end
    rates
end

function physical_domain!(pd_terms!, rates, state, transforms)

    # prepare fields in physical domain
    state_pd = Dict(s => transform_state!(t, state) for (s, t) in pairs(transforms))
    rates_pd = Dict(s => reset!(t.rates) for (s, t) in pairs(transforms))

    # compute terms in physical domain
    pd_terms!(rates_pd, state_pd)

    # transform results back to physical domain
    foldl(add_rates!, values(transforms), init=rates)
end

function add_layer!(fdlayer, transform, pdlayer)
    @assert size(pdlayer) == size(transform.pdbuffer)

    k1max = min(size(transform.fdbuffer, 1), size(fdlayer, 1)) - 1
    k2max = div(min(size(transform.fdbuffer, 2), size(fdlayer, 2)) - 1, 2)
    fft_factor = 1 / prod(size(pdlayer))

    transform.pdbuffer .= pdlayer
    mul!(transform.fdbuffer, transform.fwdplan, transform.pdbuffer)
    fdlayer[1:k1max+1, 1:k2max+1] .+= transform.fdbuffer[1:k1max+1, 1:k2max+1] * fft_factor
    fdlayer[1:k1max+1, end-k2max+1:end] .+= transform.fdbuffer[1:k1max+1, end-k2max+1:end] * fft_factor

    fdlayer
end

function add_field!(fdfield, transform, pdfield)
    for i3 = 1:size(fdfield, 3)
        add_layer!(view(fdfield, :, :, i3), transform, view(pdfield, :, :, i3))
    end
    fdfield
end

function set_layer!(fdlayer, transform, pdlayer)
    fill!(fdlayer, 0)
    add_layer!(fdlayer, transform, pdlayer)
end

function set_field!(fdfield, transform, pdfield)
    fill!(fdfield, 0)
    add_field!(fdfield, transform, pdfield)
end

function set_field!(fn::Function, fdfield, transform, domain, grid, nodes)
    n = size(transform.pdbuffer)
    x1s = x1range(domain, h1range(grid, n))
    x2s = x2range(domain, h2range(grid, n))
    x3s = x3range(domain, vrange(grid, nodes))

    for (i3, x3) = zip(1:size(fdfield, 3), x3s)
        values = (fn(x1, x2, x3) for x1=x1s, x2=x2s)
        set_layer!(view(fdfield, :, :, i3), transform, values)
    end
    fdfield
end

function get_layer!(pdlayer, transform, fdlayer)
    @assert size(pdlayer) == size(transform.pdbuffer)

    k1max = min(size(transform.fdbuffer, 1), size(fdlayer, 1)) - 1
    k2max = div(min(size(transform.fdbuffer, 2), size(fdlayer, 2)) - 1, 2)

    @views transform.fdbuffer[1:k1max+1, 1:k2max+1] .= fdlayer[1:k1max+1, 1:k2max+1]
    # .* prefactors[1:max(size(prefactors, 2), k1max+1), 1:max(size(prefactors, 2), end-k2max+1):end]
    @views transform.fdbuffer[1:k1max+1, end-k2max+1:end] .= fdlayer[1:k1max+1, end-k2max+1:end]
    # .* prefactors[1:max(size(prefactors, 2), k1max+1), max(size(prefactors, 2), end-k2max+1):end]
    transform.fdbuffer[k1max+2:end, :] .= 0
    transform.fdbuffer[:, k2max+2:end-k2max] .= 0

    mul!(transform.pdbuffer, transform.bwdplan, transform.fdbuffer)
    pdlayer .= transform.pdbuffer
end

"""
Transform a field from the frequency domain to an extended set of nodes in the
physical domain by adding extra frequencies set to zero.
"""
function get_field!(pdfield, transform, fdfield)
    for i3 = 1:equivalently(size(pdfield, 3), size(fdfield, 3))
        get_layer!(view(pdfield, :, :, i3), transform, view(fdfield, :, :, i3))
    end
    pdfield
end

# convenience function allocating a new array
function get_field(transform::HorizontalTransform{T}, fdfield, dealiasing = nothing) where T
    # we can pass a named tuple to pdsize instead of the grid,
    # since it only needs access to kmax
    k1max = size(fdfield, 1) - 1
    k2max = div(size(fdfield, 2) - 1, 2)
    n1, n2 = pdsize((k1max = k1max, k2max = k2max), dealiasing)
    n3 = size(fdfield, 3)
    get_field!(zeros(T, n1, n2, n3), transform, fdfield)
end

end # module Transform
