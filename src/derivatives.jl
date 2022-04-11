module Derivatives

using ..Helpers
using ..Grid: wavenumbers, vrange
using ..Domain: SemiperiodicDomain

function second_derivatives(grid, domain::SemiperiodicDomain{T}, nodes) where T
    k1, k2 = wavenumbers(grid)
    DD1 = reshape( - k1.^2 * (2π/convert(T, domain.hsize[1]))^2, (:, 1))
    DD2 = reshape( - k2.^2 * (2π/convert(T, domain.hsize[2]))^2, (1, :))
    DD3 = vdiff_factors(grid, domain, nodes, neighbors=true)
    (DD1=DD1, DD2=DD2, DD3=DD3)
end

function vdiff_factors(grid, domain::SemiperiodicDomain{T}, nodes; neighbors=false) where T
    α(ζ) = grid.n3global / domain.Dvmap(convert(T, ζ))
    ζ = vrange(grid, nodes)
    if neighbors
        ζnb = vrange(grid, nodes, neighbors=true)
        [(α(ζnb[i]), α(ζ[i]), α(ζnb[i+1])) for i=1:equivalently(length(ζ), length(ζnb)-1)]
    else
        collect(α.(ζ))
    end
end

end
