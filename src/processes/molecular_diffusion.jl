export MolecularDiffusion

"""
    MolecularDiffusion(field, diffusivity)

Diffusive transport of a scalar quantity ``q`` with a diffusion coefficient ``D`` that is constant in space and time.

# Arguments

- `field::Symbol`: The name of the quantity ``q``.
- `diffusivity::Real`: The diffusion coefficient ``D``.
"""
struct MolecularDiffusion <: ProcessDefinition
    field::Symbol
    diffusivity::Real
end

MolecularDiffusion(fields::NTuple{N,Symbol}, diffusivity) where N =
    Tuple(MolecularDiffusion(f, diffusivity) for f in fields)

struct DiscretizedMolecularDiffusion <: DiscretizedProcess
    field::Symbol
    diffusivity
    boundary_conditions
    derivatives
    nodes::NodeSet
end

Base.nameof(::DiscretizedMolecularDiffusion) = "Molecular Diffusion"

function init_process(diff::MolecularDiffusion, domain::Domain{T}, grid) where T
    ns = nodes(diff.field)
    bcs = init_bcs(diff.field, domain, grid)
    derivatives = second_derivatives(domain, grid, ns)
    DiscretizedMolecularDiffusion(diff.field, convert(T, diff.diffusivity),
                                  bcs, derivatives, ns)
end

state_fields(diff::DiscretizedMolecularDiffusion) = diff.field

function add_rates!(rate, term::DiscretizedMolecularDiffusion, state, t, log)
    add_laplacian!(rate[term.field], state[term.field], term.boundary_conditions...,
                   term.derivatives, term.nodes, term.diffusivity)
    rate
end

"""
Compute the Laplacian of a scalar field and add it to a different scalar field.
Both fields have to be defined on the same set of nodes. An optional prefactor
can be used to rescale the Laplacian before it is added.
"""
function add_laplacian!(scalar_output, scalar_input, lower_bc, upper_bc, df, ns, prefactor = 1)

    s_in_expanded = layers_expand_full(scalar_input, lower_bc, upper_bc, ns)
    s_out = layers(scalar_output)

    for i = 1:equivalently(length(s_in_expanded)-2, length(s_out))
        add_laplacian!(s_out[i], s_in_expanded[i:i+2], df.DD1, df.DD2, df.DD3[i], ns, prefactor)
    end

    scalar_output
end

# Laplacian in frequency domain
add_laplacian!(rhs, (vel¯, vel⁰, vel⁺)::Tuple{A1,A2,A3}, DD1, DD2, DD3, ::NodeSet, prefactor=1) where {A1,A2,A3} = # (vel¯ - 2 vel⁰ + vel⁺) / δz²
    @. rhs += prefactor * (DD3[1] * DD3[2] * vel¯ +
                           (DD1 + DD2 - DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰ +
                           DD3[2] * DD3[3] * vel⁺)
