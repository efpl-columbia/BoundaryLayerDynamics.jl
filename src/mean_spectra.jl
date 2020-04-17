# The spectra Eij(k1) and Eij(k2) are stored in two separate arrays, one for
# each horizontal direction. The first dimension corresponds to the wavenumbers
# from 0 to kmax, the second is used for the six components of the symmetric
# tensor Eij, and the third is the vertical direction. Since k1max and k2max
# can be different, the two arrays can have a different size along the first
# direction. The spectra are stored in order E11, E22, E33, E13, E23, E12. The
# `indices` tuple can be used to get the correct index for `Eij` from `i` and
# `j`.
struct MeanSpectra{T}
    Eij_k1::Array{T,3}
    Eij_k2::Array{T,3}
    indices::NTuple{3,NTuple{3,Int}}
    MeanSpectra(T, k1max, k2max, n3) = new{T}(
        zeros(T, 1+k1max, 6, n3),
        zeros(T, 1+k2max, 6, n3),
        ((1, 6, 4), (6, 2, 5), (4, 5, 3)),
    )
end

# Convenience function to get Eij(k1) and Eij(k2) for a given (i,j) and
# vertical index.
Eij(spectra::MeanSpectra, (i, j), i3) =
    (view(spectra.Eij_k1, :, spectra.indices[i][j], i3),
     view(spectra.Eij_k2, :, spectra.indices[i][j], i3))

function add_spectra!(spectra::MeanSpectra, u1::NTuple{Nh}, u2::NTuple{Nh}, u3⁺::Tuple) where Nh
    TimerOutputs.@timeit "save spectra" begin
        for i3=1:Nh
            add_Eii!(Eij(spectra, (1,1), i3), u1[i3])
            add_Eij!(Eij(spectra, (1,2), i3), u1[i3], u2[i3])
            add_Eij!(Eij(spectra, (1,3), i3), u1[i3], u3⁺[i3], u3⁺[i3+1])
            add_Eii!(Eij(spectra, (2,2), i3), u2[i3])
            add_Eij!(Eij(spectra, (2,3), i3), u2[i3], u3⁺[i3], u3⁺[i3+1])
            add_Eii!(Eij(spectra, (3,3), i3), u3⁺[i3], u3⁺[i3+1])
        end
    end
    spectra
end

function gather_spectra(spectra::MeanSpectra)
    Eij_k1 = gather_along_last_dimension(spectra.Eij_k1)
    Eij_k2 = gather_along_last_dimension(spectra.Eij_k2)
    Dict("E11_k1" => view(Eij_k1, :, spectra.indices[1][1], :),
         "E22_k1" => view(Eij_k1, :, spectra.indices[2][2], :),
         "E33_k1" => view(Eij_k1, :, spectra.indices[3][3], :),
         "E13_k1" => view(Eij_k1, :, spectra.indices[1][3], :),
         "E23_k1" => view(Eij_k1, :, spectra.indices[2][3], :),
         "E12_k1" => view(Eij_k1, :, spectra.indices[1][2], :),
         "E11_k2" => view(Eij_k2, :, spectra.indices[1][1], :),
         "E22_k2" => view(Eij_k2, :, spectra.indices[2][2], :),
         "E33_k2" => view(Eij_k2, :, spectra.indices[3][3], :),
         "E13_k2" => view(Eij_k2, :, spectra.indices[1][3], :),
         "E23_k2" => view(Eij_k2, :, spectra.indices[2][3], :),
         "E12_k2" => view(Eij_k2, :, spectra.indices[1][2], :),)
end

function reset!(spectra::MeanSpectra)
    spectra.Eij_k1 .= 0
    spectra.Eij_k2 .= 0
    spectra
end

function gather_along_last_dimension(A)
    MPI.Initialized() || return A

    # NOTE: calls to "Gatherv" need an array of Cints
    counts = MPI.Gather(Cint(length(A)), 0, MPI.COMM_WORLD)
    A_global = MPI.Gatherv(A[:], counts, 0, MPI.COMM_WORLD)

    reshape(A_global, size(A)[1:end-1]..., :)
end

zdot(z1, z2) = real(z1) * real(z2) + imag(z1) * imag(z2)
add_Eii!(Eii, ui) = add_Eij!(Eii, (i1,i2) -> abs2(ui[i1,i2]))
add_Eii!(Eii, ui¯, ui⁺) = add_Eij!(Eii, (i1,i2) -> abs2((ui¯[i1,i2] + ui⁺[i1,i2])/2))
add_Eii!(Eii, ui¯::DirichletBC, ui⁺) = add_Eij!(Eii, (i1,i2) -> abs2(((i1==i2==1 ? ui¯.value : 0) + ui⁺[i1,i2])/2))
add_Eii!(Eii, ui¯, ui⁺::DirichletBC) = add_Eij!(Eii, (i1,i2) -> abs2((ui¯[i1,i2] + (i1==i2==1 ? ui⁺.value : 0))/2))
add_Eij!(Eij, ui, uj) = add_Eij!(Eij, (i1,i2) -> zdot(ui[i1,i2], uj[i1,i2]))
add_Eij!(Eij, ui, uj¯, uj⁺) = add_Eij!(Eij, (i1,i2) -> zdot(ui[i1,i2], (uj¯[i1,i2] + uj⁺[i1,i2])/2))
add_Eij!(Eij, ui, uj¯::DirichletBC, uj⁺) = add_Eij!(Eij, (i1,i2) -> zdot(ui[i1,i2], ((i1==i2==1 ? uj¯.value : 0) + uj⁺[i1,i2])/2))
add_Eij!(Eij, ui, uj¯, uj⁺::DirichletBC) = add_Eij!(Eij, (i1,i2) -> zdot(ui[i1,i2], (uj¯[i1,i2] + (i1==i2==1 ? uj⁺.value : 0))/2))

function add_Eij!((Eij_k1, Eij_k2), Eij_of_index::Function)

    # NOTE: we assume that there are no Nyquist frequencies
    k1max = length(Eij_k1)-1 # k1 = 0, 1, …, k1max
    k2max = length(Eij_k2)-1 # k2 = 0, 1, …, k2max, -k2max, …, -1

    for i2 = 1:1+2*k2max
        k2 = i2 <= k2max+1 ? i2-1 : 2*(k2max+1)-i2
        for i1 = 1:1+k1max
            e = (i1 == 1 ? 1 : 2) * Eij_of_index(i1, i2)
            Eij_k1[i1] += e
            Eij_k2[1+k2] += e
        end
    end
    Eij_k1, Eij_k2
end
