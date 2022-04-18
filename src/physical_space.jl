module PhysicalSpace

using FFTW: FFTW
using LinearAlgebra: mul!
using ..Helpers
using ..Grids: NodeSet, vrange, nodes
using ..Domains: AbstractDomain as Domain, x1range, x2range, x3range

struct Transform2D{T,P1,P2}
    pdbuffer::Array{T,2}
    fdbuffer::Array{Complex{T},2}
    fwdplan::P1
    bwdplan::P2
    function Transform2D(::Type{T}, dims; flags=FFTW.MEASURE) where T
        # prepare FFT plans and appropriately sized buffers
        pdbuffer = zeros(T, dims)
        fdbuffer = zeros(Complex{T}, (div(dims[1],2)+1, dims[2]))
        fwdplan = FFTW.plan_rfft(pdbuffer, flags=flags)
        bwdplan = FFTW.plan_brfft(fdbuffer, dims[1], flags=flags)
        new{T,typeof(fwdplan),typeof(bwdplan)}(pdbuffer, fdbuffer, fwdplan, bwdplan)
    end
end

Base.size(transform::Transform2D, opts...) = size(transform.pdbuffer, opts...)

struct Fields{T,S,R}
    transform::T
    terms::S
    rates::R

    function Fields(::Type{T}, dims, terms, rates) where T
        transform = Transform2D(T, dims[1:2])
        new{typeof(transform),typeof(terms),typeof(rates)}(transform, terms, rates)
    end
end


# PHYSICAL SIZE FOR DEALIASING STRATEGIES ------------------

# 2D size of physical domain fields
default_size(grid) = pdsize(grid)
pdsize(grid) = pdsize(grid, nothing)
pdsize(grid, dealiasing::Nothing) = (2 + 2 * grid.k1max, 2 + 2 * grid.k2max)
pdsize(grid, dealiasing::Tuple{Int,Int}) = dealiasing
pdsize(grid, dealiasing::Symbol) = pdsize(grid, Val(dealiasing))
pdsize(grid, dealiasing::Val) = error("Unsupported dealiasing type: $(dealiasing)")
pdsize(grid, dealiasing::Val{:quadratic}) = pdsize.((grid.k1max, grid.k2max), dealiasing)
function pdsize(kmax::Integer, ::Val{:quadratic})
    # heuristics for finding a size with efficient transforms
    if isinteger(log(2, 1 + kmax)) # N is a power of 2 -> use 3/2
        3 * kmax + 3
    else # minimal size for exact products, round up if odd
        3 * kmax + (isodd(kmax) ? 1 : 2)
    end
end

# 3D size of physical domain fields
pdsize(grid, ::NodeSet{:C}, dealiasing) = (pdsize(grid, dealiasing)..., grid.n3c)
pdsize(grid, ::NodeSet{:I}, dealiasing) = (pdsize(grid, dealiasing)..., grid.n3i)
pdsize(grid, nodes::NodeSet) = pdsize(grid, nodes, nothing)

# returns a range of rational ζ-values between 0 and 1
h1range(grid, dealiasing) = LinRange(0//1, 1//1, pdsize(grid, dealiasing)[1]+1)[1:end-1]
h2range(grid, dealiasing) = LinRange(0//1, 1//1, pdsize(grid, dealiasing)[2]+1)[1:end-1]


# INITIALIZE & APPLY SETS OF TRANSFORMS --------------------

function physical_domain!(add_psrates!, rates, state, pspaces)

    # prepare fields in physical domain
    psterms = Dict(s => compute_terms!(ps, state) for (s, ps) in pairs(pspaces))
    psrates = Dict(s => reset!(ps.rates) for (s, ps) in pairs(pspaces))

    # compute rates in physical domain
    add_psrates!(psrates, psterms)

    # transform results back to physical domain
    foldl(add_rates!, values(pspaces), init=rates)
end

function init_physical_spaces((terms, rates), domain::Domain{T}, grid) where T

    # sort terms & rates by physical-domain size
    init() = (terms=Symbol[], rates=Symbol[])
    fields = Dict(pdsize(grid) => init()) # start by including default size
    for (f, s) in terms
        push!(get!(fields, s, init()).terms, f)
    end
    for (f, s) in rates
        push!(get!(fields, s, init()).rates, f)
    end

    Dict(dims => Fields(T, dims, init_terms(unique(terms), domain, grid, dims),
                                 init_rates(unique(rates), domain, grid, dims))
         for (dims, (terms, rates)) in fields)
end


# PHYSICAL-DOMAIN TERMS ------------------------------------

function init_terms(fields::Vector{Symbol}, domain::Domain{T}, grid, dims; maxit=100) where T

    terms = []
    queue = copy(fields)

    # go through fields to add their respective dependencies at the end of the queue
    for it=0:maxit

        isempty(queue) && break
        it == maxit && error("Could not initialize terms")
        field = popfirst!(queue)

        # skip fields that are already have been processed
        field in first.(terms) && continue

        # The current term can make use of the full list of terms to decide on
        # a strategy that shares inputs with other terms. We do not pass
        # indirect dependencies (i.e. the variable ‘queue’ here) to avoid
        # having the behavior depend on the order of the terms.
        term = init_term(Val(field), domain, grid, dims, terms)
        term = haskey(term, :default) ? default_term(Val(field), domain, grid, dims) : term
        haskey(term, :dependencies) && push!(queue, term.dependencies)

        # prepend array that will hold the terms
        term = NamedTuple([:values => zeros(T, pdsize(grid, nodes(field), dims)),
                           pairs(term)...])

        push!(terms, field => term)
    end

    NamedTuple(order_terms!(terms))
end

function order_terms!(terms; maxit = 100)
    i = 1
    for it=1:maxit
        i > length(terms) && break
        (field, term) = terms[i]
        deps = haskey(term, :dependencies) ? term.dependencies : ()
        # check if it has dependencies other than terms before it
        if all(dep in first.(terms[1:i-1]) for dep in deps)
            i += 1 # move on to next item
        else
            push!(terms, popat!(terms, i)) # move item to the end of the list
        end
        it == maxit && error("Could not resolve physical-domain dependencies")
    end
    terms
end

function compute_terms!(pspace, state)
    psterms = Pair[]
    for (field, term) in pairs(pspace.terms)
        # note: `compute_term!` takes the term in form of a NamedTuple but
        # returns the array with the values
        term = compute_term!(term, field, NamedTuple(psterms), state, pspace.transform)
        term isa AbstractArray || error("`compute_term!` should return the computed values")
        push!(psterms, field => term)
    end
    NamedTuple(psterms)
end

# allow calling functions with symbol of field name
init_term(field::Symbol, opts...) = init_term(Val(field), opts...)
compute_term!(term, field::Symbol, opts...) = compute_term!(term, Val(field), opts...)

# allow omitting last arguments when defining a new method
init_term(field::Val, opts...) = init_term(field, opts[1:end-1]...)
compute_term!(term, field::Val, opts...) = compute_term!(term, Val(field), opts[1:end-1]...)

# default behavior is specified with a separate function that is defined in derivatives
init_term(field::Val) = (default=true,)
function default_term end


# PHYSICAL-DOMAIN RATES ------------------------------------

function init_rates(rates::Vector{Symbol}, domain::Domain{T}, grid, dims) where T
    basefield(f) = Symbol(split(string(f), '_')[1])
    NamedTuple(field => zeros(T, pdsize(grid, nodes(basefield(field)), dims)) for field in rates)
end

function add_rates!(rates, pspaces)
    # TODO: support layers with derivatives
    for (field, values) in pairs(pspaces.rates)
        add_field!(rates[field], pspaces.transform, values)
    end
    rates
end

# APPLYING FFTS --------------------------------------------

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

set_field!(value::Real, fdfield, transform, domain, grid, nodes) =
    set_field!((x1, x2, x3) -> value, fdfield, transform, domain, grid, nodes)

function get_layer!(pdlayer, transform, fdlayer, prefactors = ones(eltype(pdlayer), (1,1)))
    @assert size(pdlayer) == size(transform.pdbuffer)

    k1max = min(size(transform.fdbuffer, 1), size(fdlayer, 1)) - 1
    k2max = div(min(size(transform.fdbuffer, 2), size(fdlayer, 2)) - 1, 2)

    @views transform.fdbuffer[1:k1max+1, 1:k2max+1] .= fdlayer[1:k1max+1, 1:k2max+1] .*
        prefactors[1:min(size(prefactors, 1), k1max+1), 1:min(size(prefactors, 2), k2max+1)]
        #prefactors[1:min(size(prefactors, 1), k1max+1), 1:min(size(prefactors, 2), end-k2max+1):end]
    @views transform.fdbuffer[1:k1max+1, end-k2max+1:end] .= fdlayer[1:k1max+1, end-k2max+1:end] .*
        prefactors[1:min(size(prefactors, 1), k1max+1), max(1, end-k2max+1):end]
        #prefactors[1:min(size(prefactors, 1), k1max+1), end+1-min(size(prefactors, 2), k2max):end]
    transform.fdbuffer[k1max+2:end, :] .= 0
    transform.fdbuffer[:, k2max+2:end-k2max] .= 0

    mul!(transform.pdbuffer, transform.bwdplan, transform.fdbuffer)
    pdlayer .= transform.pdbuffer
end

"""
Transform a field from the frequency domain to an extended set of nodes in the
physical domain by adding extra frequencies set to zero.
"""
function get_field!(pdfield, transform, fdfield, prefactors = ones(eltype(pdfield), (1,1,1)))
    for i3 = 1:equivalently(size(pdfield, 3), size(fdfield, 3))
        get_layer!(view(pdfield, :, :, i3), transform, view(fdfield, :, :, i3),
                   view(prefactors, :, :, min(size(prefactors,3),i3)))
    end
    pdfield
end

# convenience function allocating a new array
function get_field(transform::Transform2D{T}, fdfield, dealiasing = nothing) where T
    # we can pass a named tuple to pdsize instead of the grid,
    # since it only needs access to kmax
    k1max = size(fdfield, 1) - 1
    k2max = div(size(fdfield, 2) - 1, 2)
    n1, n2 = pdsize((k1max = k1max, k2max = k2max), dealiasing)
    n3 = size(fdfield, 3)
    get_field!(zeros(T, n1, n2, n3), transform, fdfield)
end




end # module PhysicalSpace
