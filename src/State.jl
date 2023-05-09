module State

using ..CBD: readcbd
using ..Domains: Domain, x1range, x2range, x3range
using ..Grids: StaggeredFourierGrid as Grid, NodeSet, nodes, vrange
using ..PhysicalSpace: get_field, set_field!, default_size, h1range, h2range
using ..BoundaryConditions: init_bcs, ConstantValue, layers_i2c

init_state(::Type{T}, grid, fields) where T =
    NamedTuple(f => zeros(T, grid, nodes(f)) for f in fields)

function initialize!(state, domain, grid, physical_spaces;
                     add_noise = false, initial_conditions...)
    for field in keys(initial_conditions)
        field in keys(state) || error("Cannot initialize `$field`: unknown field")
    end
    for (field, values) in pairs(state)
        if field in keys(initial_conditions)
            set_field!(initial_conditions[field], values,
                       physical_spaces[default_size(grid)].transform,
                       domain, grid, nodes(field))
            add_noise && add_noise!(values)
        else
            fill!(values, 0)
        end
    end
    state
end

function initialize!(state, path, domain, grid, physical_spaces)
    for field in keys(state)
        fpath = joinpath(path, "$field.cbd")

        if !isfile(fpath)
            @warn "No initial values found for `$field`, set to zero."
            state[field] .= 0
            continue
        end

        # (xmin, xmax, x1, x2, x3, pfield)
        cbd = readcbd(fpath, collect(x3range(domain, vrange(grid, nodes(field)))), grid.comm)

        # ensure that file data is compatible with current simulation
        xmin, xmax = extrema(domain)
        @assert all(cbd[1] .≈ xmin)
        @assert all(cbd[2] .≈ xmax)

        x1, x2, x3 = cbd[3:5]
        centered = (x1[1] != 0)

        # check if x1- & x2-values are consistent
        @assert all(x1 .≈ LinRange(xmin[1], xmax[1], 2*length(x1)+1)[(centered ? 2 : 1):2:end-1])
        @assert all(x2 .≈ LinRange(xmin[2], xmax[2], 2*length(x2)+1)[(centered ? 2 : 1):2:end-1])

        # TODO: allow interpolating x3-values
        @assert all(cbd[5] .≈ x3range(domain, vrange(grid, nodes(field))))

        # the snapshot might be any size, not just those with initialized transforms
        pdims = size(cbd[end])[1:2]
        transform = haskey(physical_spaces, pdims) ? physical_spaces[pdims].transform : Transform2D(T, pdims)
        set_field!(state[field], transform, cbd[end], centered = centered)
    end
end

reset!(state) = (fill!.(values(state), 0); state)

function add_noise!(vel::AbstractArray{Complex{T},3}, intensity::T = one(T) / 8) where T
    intensity == 0 && return vel
    for k=1:size(vel, 3)
        σ = real(vel[1,1,k]) * intensity
        for j=1:size(vel, 2), i=(j == 1 ? 2 : 1):size(vel, 1) # do not modify mean flow
            vel[i,j,k] += σ * randn(Complex{T})
        end
    end
    vel
end

function getterm(state, term, domain::Domain{T}, grid, physical_spaces, ns = nodes(term)) where T
    pdims = default_size(grid)
    field = get_field(physical_spaces[pdims].transform, state[term])
    ns == nodes(term) ? field : interpolate(field, term, domain, grid)
end

function interpolate(field, term, domain::Domain{T}, grid) where T
    pdims = size(field)[1:2]
    field = layers_i2c(field, init_bcs(term, domain, grid, pdims)...)
    interpolated = zeros(T, pdims..., length(field)-1)
    for i3=1:size(interpolated, 3)
        interpolate!(view(interpolated, :, :, i3), field[i3:i3+1]...)
    end
    interpolated
end


interpolate!(mid, below, above) = @. mid = (below + above) / 2
interpolate!(mid::AbstractArray{T}, below::ConstantValue, above) where {T<:Real} =
    @. mid = (below.value + above) / 2
interpolate!(mid::AbstractArray{T}, below, above::ConstantValue) where {T<:Real} =
    @. mid = (below + above.value) / 2

coordinates(domain, grid) = coordinates(domain, grid, :vel1)
coordinates(domain, grid, dim::Int) = coordinates(domain, grid, :vel1, Val(dim))
coordinates(domain, grid, field, dim::Int) = coordinates(domain, grid, field, Val(dim))
coordinates(domain, grid, field, ::Val{1}) = x1range(domain, h1range(grid, default_size(grid)))
coordinates(domain, grid, field, ::Val{2}) = x2range(domain, h2range(grid, default_size(grid)))
coordinates(domain, grid, field, ::Val{3}) = x3range(domain, vrange(grid, nodes(field)))
coordinates(domain, grid, field::Symbol) = begin
    ((x1, x2, x3) for x1=coordinates(domain, grid, field, Val(1)),
                      x2=coordinates(domain, grid, field, Val(2)),
                      x3=coordinates(domain, grid, field, Val(3)))
end

end # module State
