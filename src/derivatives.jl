function vdiff_factors(gd, gm::GridMapping{T}, ns; neighbors=false) where T
    α(ζ) = gd.nz_global / gm.Dvmap(ζ)
    ζ = vrange(T, gd, ns)
    if neighbors
        ζnb = vrange(T, gd, ns, neighbors=true)
        [(α(ζnb[i]), α(ζ[i]), α(ζnb[i+1])) for i=1:equivalently(length(ζ), length(ζnb)-1)]
    else
        collect(α.(ζ))
    end
end

struct DerivativeFactors{T<:SupportedReals}
    D1::Array{Complex{T},2}
    D2::Array{Complex{T},2}
    D3_h::Array{T,1}
    D3_v::Array{T,1}
    DD1::Array{T,2}
    DD2::Array{T,2}
    DD3_h::Array{Tuple{T,T,T},1}
    DD3_v::Array{Tuple{T,T,T},1}
    DerivativeFactors(gd::DistributedGrid, gm::GridMapping{T}) where T = begin
        k1, k2 = wavenumbers(gd)
        new{T}(
            reshape(1im * k1 * (2π/gm.hsize1), (:, 1)),
            reshape(1im * k2 * (2π/gm.hsize2), (1, :)),
            vdiff_factors(gd, gm, NodeSet(:H)),
            vdiff_factors(gd, gm, NodeSet(:V)),
            reshape( - k1.^2 * (2π/gm.hsize1)^2, (:, 1)),
            reshape( - k2.^2 * (2π/gm.hsize2)^2, (1, :)),
            vdiff_factors(gd, gm, NodeSet(:H), neighbors=true),
            vdiff_factors(gd, gm, NodeSet(:V), neighbors=true),
    )
    end
end

get_D3(df, ::NodeSet{:H}) = df.D3_h
get_D3(df, ::NodeSet{:V}) = df.D3_v
get_DD3(df, ::NodeSet{:H}) = df.DD3_h
get_DD3(df, ::NodeSet{:V}) = df.DD3_v
Broadcast.broadcastable(df::DerivativeFactors) = Ref(df) # to use as argument of elementwise functions

get_field_dx1!(field_dx1_pd, fields_fd, ht, df, ns) =
    get_field!(field_dx1_pd, ht, fields_fd, df.D1, ns)
get_field_dx2!(field_dx2_pd, fields_fd, ht, df, ns) =
    get_field!(field_dx2_pd, ht, fields_fd, df.D2, ns)

# Horizontal derivatives in frequency domain
add_derivative!(f_out, f_in::AbstractArray, D, prefactor = 1) = @. f_out += prefactor * D * f_in

# Vertical derivatives in frequency or physical domain
add_derivative!(f_out, (f_in¯, f_in⁺)::Tuple, D, prefactor = 1) =
        @. f_out += prefactor * (-D * f_in¯ + D * f_in⁺)

# Boundary conditions of vertical derivatives in physical domain only
add_derivative!(f_out, (lbc, f_in⁺)::Tuple{SolidWallBC,A}, D, prefactor = 1) where {A} =
        @. f_out += prefactor * (-D * lbc.value + D * f_in⁺)
add_derivative!(f_out, (f_in¯, ubc)::Tuple{A,SolidWallBC}, D, prefactor = 1) where {A} =
        @. f_out += prefactor * (-D * f_in¯ + D * ubc.value)

# Laplacian in frequency domain
add_laplacian!(rhs, (vel¯, vel⁰, vel⁺)::Tuple{A1,A2,A3}, DD1, DD2, DD3, ::NodeSet, prefactor=1) where {A1,A2,A3} = # (vel¯ - 2 vel⁰ + vel⁺) / δz²
    @. rhs += prefactor * (DD3[1] * DD3[2] * vel¯ +
                           (DD1 + DD2 - DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰ +
                           DD3[2] * DD3[3] * vel⁺)
add_laplacian!(rhs, (lbc, vel⁰, vel⁺)::Tuple{NeumannBC,A2,A3}, DD1, DD2, DD3, ::NodeSet{:H}, prefactor=1) where {A2,A3} = # (- δz * LBC - vel⁰ + vel⁺) / δz²
    (@. rhs += prefactor * ((DD1 + DD2 - DD3[2] * DD3[3]) * vel⁰ + DD3[2] * DD3[3] * vel⁺);
     rhs[1,1] -= prefactor * DD3[2] * lbc.gradient; rhs)
add_laplacian!(rhs, (vel¯, vel⁰, ubc)::Tuple{A1,A2,NeumannBC}, DD1, DD2, DD3, ::NodeSet{:H}, prefactor=1) where {A1,A2} = # (vel¯ - vel⁰ + δz * UBC) / δz²
    (@. rhs += prefactor * (DD3[1] * DD3[2] * vel¯ + (DD1 + DD2 - DD3[1] * DD3[2]) * vel⁰);
     rhs[1,1] += prefactor * DD3[2] * ubc.gradient; rhs)
add_laplacian!(rhs, (lbc, vel⁰, vel⁺)::Tuple{SolidWallBC,A2,A3}, DD1, DD2, DD3, ::NodeSet{:H}, prefactor=1) where {A2,A3} = # (8/3 lbc - 4 vel⁰ + 4/3 vel⁺) / δz²
    (@. rhs += prefactor * ((DD1 + DD2 - 9/3 * DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰ +
                            (1/3 * DD3[1] * DD3[2] + DD3[2] * DD3[3]) * vel⁺);
     rhs[1,1] += prefactor * 8/3 * DD3[1] * DD3[2] * lbc.value; rhs)
add_laplacian!(rhs, (vel¯, vel⁰, ubc)::Tuple{A1,A2,SolidWallBC}, DD1, DD2, DD3, ::NodeSet{:H}, prefactor=1) where {A1,A2} = # (4/3 vel¯ - 4 vel⁰ + 8/3 ubc) / δz²
    (@. rhs += prefactor * ((DD3[1] * DD3[2] + 1/3 * DD3[2] * DD3[3]) * vel¯ +
                            (DD1 + DD2 - DD3[1] * DD3[2] - 9/3 * DD3[2] * DD3[3]) * vel⁰);
     rhs[1,1] += prefactor * 8/3 * DD3[2] * DD3[3] * ubc.value; rhs)
add_laplacian!(rhs, (lbc, vel⁰, vel⁺)::Tuple{SolidWallBC,A2,A3}, DD1, DD2, DD3, ::NodeSet{:V}, prefactor=1) where {A2,A3} = # (lbc - 2 vel⁰ + vel⁺) / δz²
    (@. rhs += prefactor * ((DD1 + DD2 - DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰ +
                            DD3[2] * DD3[3] * vel⁺);
     rhs[1,1] += prefactor * DD3[1] * DD3[2] * lbc.value; rhs)
add_laplacian!(rhs, (vel¯, vel⁰, ubc)::Tuple{A1,A2,SolidWallBC}, DD1, DD2, DD3, ::NodeSet{:V}, prefactor=1) where {A1,A2} = # (vel¯ - 2 vel⁰ + ubc) / δz²
    (@. rhs += prefactor * (DD3[1] * DD3[2] * vel¯ +
                            (DD1 + DD2 - DD3[1] * DD3[2] - DD3[2] * DD3[3]) * vel⁰);
     rhs[1,1] += prefactor * DD3[2] * DD3[3] * ubc.value; rhs)

# Vorticity in frequency domain → no boundary conditions required!
@inline set_vorticity_1!(rot1, (u2¯, u2⁺), u3, D2, D3) = @. rot1 = D2 * u3 - (-D3 * u2¯ + D3 * u2⁺)
@inline set_vorticity_2!(rot2, (u1¯, u1⁺), u3, D1, D3) = @. rot2 = (-D3 * u1¯ + D3 * u1⁺) - D1 * u3
@inline set_vorticity_3!(rot3, u1, u2, D1, D2) = @. rot3 = D1 * u2 - D2 * u1

# Divergence in frequency domain
# Note: The divergence functions allow for rescaling the horizontal
# contributions with a (horizontally) constant value. This is used to run the
# pressure solver on the system (α^C¯¹ DG) p = α^C¯¹ (D u + b_c), for which the
# tridiagonal matrix is symmetric.
div!(div, u1, u2, (u3¯, u3⁺), D1, D2, D3, hfactor=1) =
    (@. div = u1 * D1 * hfactor + u2 * D2 * hfactor - D3 * u3¯ + D3 * u3⁺; div)
div!(div::AbstractArray{T}, u1, u2, (u3¯, u3⁺)::Tuple{DirichletBC,A}, D1, D2, D3, hfactor=1) where {T <: Complex, A} =
(@. div = u1 * D1 * hfactor + u2 * D2 * hfactor + D3 * u3⁺; div[1,1] -= D3 * u3¯.value; div)
div!(div::AbstractArray{T}, u1, u2, (u3¯, u3⁺)::Tuple{A,DirichletBC}, D1, D2, D3, hfactor=1) where {T <: Complex, A} =
    (@. div = u1 * D1 * hfactor + u2 * D2 * hfactor - D3 * u3¯; div[1,1] += D3 * u3⁺.value; div)

function add_derivative_x3!(scalar_output, scalar_input, lower_bc, upper_bc, df, ns)

    s_out = layers(scalar_output)
    s_in_expanded = layers_expand_half(scalar_input, lower_bc, upper_bc, ns)

    # `ns` is are the nodes of the input, we need D3 for the nodes of the output
    D3 = ns isa NodeSet{:H} ? df.D3_v : ns isa NodeSet{:V} ? df.D3_h : error("Invalid node set")

    for i in 1:equivalently(length(s_out), length(s_in_expanded)-1)
        add_derivative!(s_out[i], s_in_expanded[i:i+1], D3[i])
    end

    scalar_output
end

"""
Compute the gradient of a scalar field and add it to a vector field. The scalar
field and the horizontal components of the vector field are defined on C-nodes
while the vertical component of the vector field are defined on I-nodes. An
optional prefactor can be used to rescale the gradient before it is added.
"""
function add_gradient!(vector_output, scalar_input, bc::UnspecifiedBC, df, prefactor = 1)

    v1 = layers(vector_output[1])
    v2 = layers(vector_output[2])
    v3 = layers(vector_output[3])

    s = layers(scalar_input)
    s_expanded = layers_expand_c_to_i(scalar_input, bc)

    add_derivative!.(v1, s, (df.D1, ), prefactor)
    add_derivative!.(v2, s, (df.D2, ), prefactor)
    for i = 1:equivalently(length(v3), length(s_expanded)-1)
        add_derivative!(v3[i], s_expanded[i:i+1], df.D3_v[i], prefactor)
    end

    vector_output
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
        add_laplacian!(s_out[i], s_in_expanded[i:i+2], df.DD1, df.DD2, get_DD3(df, ns)[i], ns, prefactor)
    end

    scalar_output
end

"""
Compute the vorticity of a vector field and write it to a vector field. The
horizontal components of the input and the vertical component of the output are
defined on C-nodes while the vertical component of the input and the horizontal
component of the output are defined on I-nodes.
"""
function set_vorticity!(vector_output, vector_input, lower_bcs, upper_bcs, df)

    v_out = layers.(vector_output)
    v_in = layers.(vector_input)
    v_in_expanded = layers_expand_c_to_i.(vector_input[1:2], upper_bcs[1:2])

    for i=1:equivalently(length(v_out[1]), length(v_in_expanded[2])-1, length(v_in[3]))
        set_vorticity_1!(v_out[1][i], v_in_expanded[2][i:i+1], v_in[3][i], df.D2, df.D3_v[i])
    end

    for i=1:equivalently(length(v_out[2]), length(v_in_expanded[1])-1, length(v_in[3]))
        set_vorticity_2!(v_out[2][i], v_in_expanded[1][i:i+1], v_in[3][i], df.D1, df.D3_v[i])
    end

    for i=1:equivalently(length(v_out[3]), length(v_in[1]), length(v_in[2]))
        set_vorticity_3!(v_out[3][i], v_in[1][i], v_in[2][i], df.D1, df.D2)
    end

    vector_output
end
