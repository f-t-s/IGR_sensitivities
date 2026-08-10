# ============================================================
# 2D uniform doubly-periodic Cartesian mesh
# ============================================================

export CartesianMesh2D

"""
    CartesianMesh2D

Uniform Cartesian grid of `N_ex × N_ey` axis-aligned quadrilateral
elements on the doubly-periodic domain `[0, Lx] × [0, Ly]`.

The grid is regular, so the reference-to-physical map on every element
is affine and separable: `x = x_left + Jx·(1+ξ)`, `y = y_left + Jy·(1+η)`
with constant directional Jacobians `Jx = Δx/2`, `Jy = Δy/2`.

Node coordinates are stored as 4D arrays `(n_p, n_p, N_ex, N_ey)` indexed
`[i, j, ex, ey]` — `i` is the x-node, `j` the y-node, matching the layout
of every state field in this package.
"""
struct CartesianMesh2D
    N_ex::Int               # number of elements in x
    N_ey::Int               # number of elements in y
    Lx::Float64             # domain length in x
    Ly::Float64             # domain length in y
    Δx::Float64             # element width in x
    Δy::Float64             # element width in y
    Jx::Float64             # x Jacobian = Δx / 2
    Jy::Float64             # y Jacobian = Δy / 2
    x::Array{Float64,4}     # node x-coordinates, (n_p, n_p, N_ex, N_ey)
    y::Array{Float64,4}     # node y-coordinates, (n_p, n_p, N_ex, N_ey)
end

"""
    CartesianMesh2D(N_ex, N_ey, Lx, Ly, basis)

Construct a uniform doubly-periodic mesh on `[0, Lx] × [0, Ly]`.
Tensor-product LGL nodes are taken from `basis.ξ`.
"""
function CartesianMesh2D(N_ex::Int, N_ey::Int, Lx::Float64, Ly::Float64, basis::DGBasis)
    Δx = Lx / N_ex
    Δy = Ly / N_ey
    Jx = Δx / 2.0
    Jy = Δy / 2.0
    n_p = basis.p + 1
    x = zeros(n_p, n_p, N_ex, N_ey)
    y = zeros(n_p, n_p, N_ex, N_ey)
    for ey in 1:N_ey, ex in 1:N_ex
        x_left = (ex - 1) * Δx
        y_left = (ey - 1) * Δy
        for j in 1:n_p, i in 1:n_p
            x[i, j, ex, ey] = x_left + Jx * (1.0 + basis.ξ[i])
            y[i, j, ex, ey] = y_left + Jy * (1.0 + basis.ξ[j])
        end
    end
    return CartesianMesh2D(N_ex, N_ey, Lx, Ly, Δx, Δy, Jx, Jy, x, y)
end
