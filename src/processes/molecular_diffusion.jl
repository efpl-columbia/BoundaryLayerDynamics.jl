export MolecularDiffusion

struct MolecularDiffusion <: ProcessDefinition
    field::Symbol
    diffusivity
end

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

    s_in_expanded = layers_expand_full(scalar_input, lower_bc, upper_bc)
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
add_laplacian!(rhs, (lbc, vel⁰, vel⁺)::Tuple{ConstantGradient,A2,A3}, DD1, DD2, DD3, ::NodeSet{:C}, prefactor=1) where {A2,A3} = # (- δz * LBC - vel⁰ + vel⁺) / δz²
    (@. rhs += prefactor * ((DD1 + DD2 - DD3[2] * DD3[3]) * vel⁰ + DD3[2] * DD3[3] * vel⁺);
     rhs[1,1] -= prefactor * DD3[2] * lbc.gradient; rhs)
add_laplacian!(rhs, (vel¯, vel⁰, ubc)::Tuple{A1,A2,ConstantGradient}, DD1, DD2, DD3, ::NodeSet{:C}, prefactor=1) where {A1,A2} = # (vel¯ - vel⁰ + δz * UBC) / δz²
    (@. rhs += prefactor * (DD3[1] * DD3[2] * vel¯ + (DD1 + DD2 - DD3[1] * DD3[2]) * vel⁰);
     rhs[1,1] += prefactor * DD3[2] * ubc.gradient; rhs)
add_laplacian!(rhs, (lbc, vel⁰, vel⁺)::Tuple{ConstantValue,A2,A3}, DD1, DD2, DD3, ::NodeSet{:C}, prefactor=1) where {A2,A3} = # (8/3 lbc - 4 vel⁰ + 4/3 vel⁺) / δz²
    (@. rhs += prefactor * ((DD1 + DD2 - 9/3 * DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰ +
                            (1/3 * DD3[1] * DD3[2] + DD3[2] * DD3[3]) * vel⁺);
     rhs[1,1] += prefactor * 8/3 * DD3[1] * DD3[2] * lbc.value; rhs)
add_laplacian!(rhs, (vel¯, vel⁰, ubc)::Tuple{A1,A2,ConstantValue}, DD1, DD2, DD3, ::NodeSet{:C}, prefactor=1) where {A1,A2} = # (4/3 vel¯ - 4 vel⁰ + 8/3 ubc) / δz²
    (@. rhs += prefactor * ((DD3[1] * DD3[2] + 1/3 * DD3[2] * DD3[3]) * vel¯ +
                            (DD1 + DD2 - DD3[1] * DD3[2] - 9/3 * DD3[2] * DD3[3]) * vel⁰);
     rhs[1,1] += prefactor * 8/3 * DD3[2] * DD3[3] * ubc.value; rhs)
add_laplacian!(rhs, (lbc, vel⁰, vel⁺)::Tuple{ConstantValue,A2,A3}, DD1, DD2, DD3, ::NodeSet{:I}, prefactor=1) where {A2,A3} = # (lbc - 2 vel⁰ + vel⁺) / δz²
    (@. rhs += prefactor * ((DD1 + DD2 - DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰ +
                            DD3[2] * DD3[3] * vel⁺);
     rhs[1,1] += prefactor * DD3[1] * DD3[2] * lbc.value; rhs)
add_laplacian!(rhs, (vel¯, vel⁰, ubc)::Tuple{A1,A2,ConstantValue}, DD1, DD2, DD3, ::NodeSet{:I}, prefactor=1) where {A1,A2} = # (vel¯ - 2 vel⁰ + ubc) / δz²
    (@. rhs += prefactor * (DD3[1] * DD3[2] * vel¯ +
                            (DD1 + DD2 - DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰);
     rhs[1,1] += prefactor * DD3[2] * DD3[3] * ubc.value; rhs)
