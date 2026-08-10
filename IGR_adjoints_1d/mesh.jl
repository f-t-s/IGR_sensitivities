# ============================================================
# 1D uniform periodic mesh
# ============================================================

export PeriodicMesh1D

struct PeriodicMesh1D
    N_e::Int              # number of elements
    L::Float64            # domain length [0, L]
    Δx::Float64           # element width
    J::Float64            # Jacobian = Δx / 2 (reference [-1,1] → physical)
    x::Matrix{Float64}    # node coordinates, (p+1) × N_e
end

"""
    PeriodicMesh1D(N_e, L, basis)

Construct a uniform periodic mesh on [0, L] with `N_e` elements.
Node coordinates are computed from the LGL nodes in `basis`.
"""
function PeriodicMesh1D(N_e::Int, L::Float64, basis::DGBasis)
    Δx = L / N_e
    J = Δx / 2.0
    n_p = basis.p + 1
    x = zeros(n_p, N_e)
    for e in 1:N_e
        x_left = (e - 1) * Δx
        for i in 1:n_p
            x[i, e] = x_left + J * (1.0 + basis.ξ[i])
        end
    end
    return PeriodicMesh1D(N_e, L, Δx, J, x)
end
