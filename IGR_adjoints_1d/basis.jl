# ============================================================
# DG basis: LGL nodes, quadrature weights, differentiation matrix
# ============================================================

export DGBasis

struct DGBasis
    p::Int              # polynomial degree
    ξ::Vector{Float64}  # LGL nodes on [-1,1], length p+1
    w::Vector{Float64}  # LGL quadrature weights, length p+1
    D::Matrix{Float64}  # differentiation matrix (p+1)×(p+1)
end

"""
    barycentric_weights(ξ)

Compute barycentric weights for the nodes `ξ`.
"""
function barycentric_weights(ξ)
    n = length(ξ)
    λ = ones(n)
    for j in 1:n
        for i in 1:n
            i == j && continue
            λ[j] *= (ξ[j] - ξ[i])
        end
        λ[j] = 1.0 / λ[j]
    end
    return λ
end

"""
    differentiation_matrix(ξ, λ)

Compute the polynomial differentiation matrix at nodes `ξ`
with barycentric weights `λ`, using the standard formula.
"""
function differentiation_matrix(ξ, λ)
    n = length(ξ)
    D = zeros(n, n)
    for j in 1:n
        for i in 1:n
            i == j && continue
            D[i, j] = (λ[j] / λ[i]) / (ξ[i] - ξ[j])
            D[i, i] -= D[i, j]
        end
    end
    return D
end

"""
    DGBasis(p)

Construct a DG basis of polynomial degree `p` using LGL nodes.
"""
function DGBasis(p::Int)
    if p == 0
        # Single midpoint node: finite-volume-like DG
        ξ = [0.0]
        w = [2.0]
        D = zeros(1, 1)
    else
        ξ, w = gausslobatto(p + 1)
        λ = barycentric_weights(ξ)
        D = differentiation_matrix(ξ, λ)
    end
    return DGBasis(p, ξ, w, D)
end
