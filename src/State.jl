module State

using ..Domains: ABLDomain as Domain, x1range, x2range, x3range
using ..Grids: StaggeredFourierGrid as Grid, NodeSet, nodes, vrange
using ..PhysicalSpace: get_field, set_field!, default_size, h1range, h2range
using ..BoundaryConditions: init_bcs, ConstantValue, layers_i2c
using ..Processes: state_fields

init_state(::Type{T}, grid, processes) where T =
    NamedTuple(f => zeros(T, grid, nodes(f)) for f in state_fields(processes))

function initialize!(state, domain, grid, physical_spaces; initial_conditions...)
    # TODO: consider setting all other fields to zero
    for (field, ic) in initial_conditions
        set_field!(ic, state[field], physical_spaces[default_size(grid)].transform,
                   domain, grid, nodes(field))
    end
end

reset!(state) = (fill!.(values(state), 0); state)

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
        interpolate!(interpolated[:,:,i3], field[i3:i3+1]...)
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
