# ============================================================
# Enzyme AD wrappers for the 2D IGR Euler solver.
#
# Uses Enzyme reverse mode for efficient scalar-objective gradients
# with respect to the initial condition.
#
# Key workarounds (carried over from the 1D package, Enzyme v0.13 +
# Julia 1.12):
#   1. The SSP-RK3 loop is inlined in each objective function rather
#      than calling `run_forward!`, which triggers an opaque-pointer
#      bug at the augmented-forward / reverse-pass boundary.
#   2. Objectives are top-level functions (not closures) with explicit
#      Const/Active argument annotations.
#   3. set_runtime_activity is used for struct field access.
#   4. No keyword arguments anywhere in the differentiated call chain.
# ============================================================

export enzyme_adjoint_ic

# ============================================================
# Gaussian window shared by the :windowed_kinetic_energy objective
# (PDE terminal condition in adjoint_pde.jl and the Enzyme objective
# below):  J = ∫ w(x,y) |μ|²/(2ρ) dxdy,
#          w(x,y) = exp(-((x-xc)² + (y-yc)²)/σ²)
# ============================================================
const KE_WINDOW = (xc = 0.55, yc = 0.50, σ = 0.15)
export KE_WINDOW

@inline ke_window(x, y) =
    exp(-((x - KE_WINDOW.xc)^2 + (y - KE_WINDOW.yc)^2) / KE_WINDOW.σ^2)
export ke_window

# ============================================================
# Top-level objective functions for Enzyme reverse mode.
# Signature (shared):
#   (α, μx0, μy0, ρ0, E0, n_p, N_ex, N_ey, n_steps, Δt,
#    basis, mesh, γ, n_iter, solver)
# ============================================================

# Inlined SSP-RK3 forward integration. Returns the final state arrays.
# Kept as a documented block; copy-inlined into each objective below.

"""
    _enzyme_objective(α, μx0, μy0, ρ0, E0, ...)

Objective J = ∫ ρ(x,T)² dx for Enzyme reverse-mode differentiation.
"""
function _enzyme_objective(α, μx0, μy0, ρ0, E0, n_p, N_ex, N_ey, n_steps, Δt,
                           basis, mesh, γ, n_iter, solver)
    μx = copy(μx0); μy = copy(μy0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, n_p, N_ex, N_ey)
    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)

    for _ in 1:n_steps
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx1 = μx + Δt*dμx; @. μy1 = μy + Δt*dμy
        @. ρ1 = ρ + Δt*dρ;    @. E1 = E + Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx2 = 3/4*μx + 1/4*μx1 + 1/4*Δt*dμx
        @. μy2 = 3/4*μy + 1/4*μy1 + 1/4*Δt*dμy
        @. ρ2  = 3/4*ρ  + 1/4*ρ1  + 1/4*Δt*dρ
        @. E2  = 3/4*E  + 1/4*E1  + 1/4*Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx = 1/3*μx + 2/3*μx2 + 2/3*Δt*dμx
        @. μy = 1/3*μy + 2/3*μy2 + 2/3*Δt*dμy
        @. ρ  = 1/3*ρ  + 2/3*ρ2  + 2/3*Δt*dρ
        @. E  = 1/3*E  + 2/3*E2  + 2/3*Δt*dE
    end

    J_val = zero(eltype(ρ))
    for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
        J_val += basis.w[i] * basis.w[j] * mesh.Jx * mesh.Jy * ρ[i,j,ex,ey]^2
    end
    return J_val
end

"""
    _enzyme_objective_l2_pressure(α, μx0, μy0, ρ0, E0, ...)

Objective J = ∫ P(x,T)² dx.
"""
function _enzyme_objective_l2_pressure(α, μx0, μy0, ρ0, E0, n_p, N_ex, N_ey, n_steps, Δt,
                                       basis, mesh, γ, n_iter, solver)
    μx = copy(μx0); μy = copy(μy0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, n_p, N_ex, N_ey)
    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)

    for _ in 1:n_steps
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx1 = μx + Δt*dμx; @. μy1 = μy + Δt*dμy
        @. ρ1 = ρ + Δt*dρ;    @. E1 = E + Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx2 = 3/4*μx + 1/4*μx1 + 1/4*Δt*dμx
        @. μy2 = 3/4*μy + 1/4*μy1 + 1/4*Δt*dμy
        @. ρ2  = 3/4*ρ  + 1/4*ρ1  + 1/4*Δt*dρ
        @. E2  = 3/4*E  + 1/4*E1  + 1/4*Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx = 1/3*μx + 2/3*μx2 + 2/3*Δt*dμx
        @. μy = 1/3*μy + 2/3*μy2 + 2/3*Δt*dμy
        @. ρ  = 1/3*ρ  + 2/3*ρ2  + 2/3*Δt*dρ
        @. E  = 1/3*E  + 2/3*E2  + 2/3*Δt*dE
    end

    J_val = zero(eltype(ρ))
    for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
        P = (γ-1) * (E[i,j,ex,ey] - (μx[i,j,ex,ey]^2 + μy[i,j,ex,ey]^2)/(2*ρ[i,j,ex,ey]))
        J_val += basis.w[i] * basis.w[j] * mesh.Jx * mesh.Jy * P^2
    end
    return J_val
end

"""
    _enzyme_objective_kinetic_energy(α, μx0, μy0, ρ0, E0, ...)

Objective J = (∫ |μ|²/(2ρ) dx) / (∫ ρ dx) — mass-averaged kinetic energy.
"""
function _enzyme_objective_kinetic_energy(α, μx0, μy0, ρ0, E0, n_p, N_ex, N_ey, n_steps, Δt,
                                          basis, mesh, γ, n_iter, solver)
    μx = copy(μx0); μy = copy(μy0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, n_p, N_ex, N_ey)
    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)

    for _ in 1:n_steps
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx1 = μx + Δt*dμx; @. μy1 = μy + Δt*dμy
        @. ρ1 = ρ + Δt*dρ;    @. E1 = E + Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx2 = 3/4*μx + 1/4*μx1 + 1/4*Δt*dμx
        @. μy2 = 3/4*μy + 1/4*μy1 + 1/4*Δt*dμy
        @. ρ2  = 3/4*ρ  + 1/4*ρ1  + 1/4*Δt*dρ
        @. E2  = 3/4*E  + 1/4*E1  + 1/4*Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx = 1/3*μx + 2/3*μx2 + 2/3*Δt*dμx
        @. μy = 1/3*μy + 2/3*μy2 + 2/3*Δt*dμy
        @. ρ  = 1/3*ρ  + 2/3*ρ2  + 2/3*Δt*dρ
        @. E  = 1/3*E  + 2/3*E2  + 2/3*Δt*dE
    end

    KE = zero(eltype(ρ)); M = zero(eltype(ρ))
    for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
        wJ = basis.w[i] * basis.w[j] * mesh.Jx * mesh.Jy
        KE += wJ * (μx[i,j,ex,ey]^2 + μy[i,j,ex,ey]^2) / (2*ρ[i,j,ex,ey])
        M  += wJ * ρ[i,j,ex,ey]
    end
    return KE / M
end

"""
    _enzyme_objective_windowed_kinetic_energy(α, μx0, μy0, ρ0, E0, ...)

Objective J = ∫ w(x,y) |μ|²/(2ρ) dxdy at t = T — kinetic energy in the
Gaussian window `KE_WINDOW` (not mass-normalized).
"""
function _enzyme_objective_windowed_kinetic_energy(α, μx0, μy0, ρ0, E0, n_p, N_ex, N_ey, n_steps, Δt,
                                                   basis, mesh, γ, n_iter, solver)
    μx = copy(μx0); μy = copy(μy0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, n_p, N_ex, N_ey)
    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)

    for _ in 1:n_steps
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx1 = μx + Δt*dμx; @. μy1 = μy + Δt*dμy
        @. ρ1 = ρ + Δt*dρ;    @. E1 = E + Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx2 = 3/4*μx + 1/4*μx1 + 1/4*Δt*dμx
        @. μy2 = 3/4*μy + 1/4*μy1 + 1/4*Δt*dμy
        @. ρ2  = 3/4*ρ  + 1/4*ρ1  + 1/4*Δt*dρ
        @. E2  = 3/4*E  + 1/4*E1  + 1/4*Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx = 1/3*μx + 2/3*μx2 + 2/3*Δt*dμx
        @. μy = 1/3*μy + 2/3*μy2 + 2/3*Δt*dμy
        @. ρ  = 1/3*ρ  + 2/3*ρ2  + 2/3*Δt*dρ
        @. E  = 1/3*E  + 2/3*E2  + 2/3*Δt*dE
    end

    J_val = zero(eltype(ρ))
    for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
        w = ke_window(mesh.x[i,j,ex,ey], mesh.y[i,j,ex,ey])
        wJ = basis.w[i] * basis.w[j] * mesh.Jx * mesh.Jy
        J_val += wJ * w * (μx[i,j,ex,ey]^2 + μy[i,j,ex,ey]^2) / (2*ρ[i,j,ex,ey])
    end
    return J_val
end

"""
    _enzyme_objective_weighted_momentum(α, μx0, μy0, ρ0, E0, ...)

Objective J = ∫ μx(x,T) sin(2πx/Lx) sin(2πy/Ly) dx — a smooth linear
functional of the final momentum field.
"""
function _enzyme_objective_weighted_momentum(α, μx0, μy0, ρ0, E0, n_p, N_ex, N_ey, n_steps, Δt,
                                             basis, mesh, γ, n_iter, solver)
    μx = copy(μx0); μy = copy(μy0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, n_p, N_ex, N_ey)
    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)

    for _ in 1:n_steps
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx1 = μx + Δt*dμx; @. μy1 = μy + Δt*dμy
        @. ρ1 = ρ + Δt*dρ;    @. E1 = E + Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx2 = 3/4*μx + 1/4*μx1 + 1/4*Δt*dμx
        @. μy2 = 3/4*μy + 1/4*μy1 + 1/4*Δt*dμy
        @. ρ2  = 3/4*ρ  + 1/4*ρ1  + 1/4*Δt*dρ
        @. E2  = 3/4*E  + 1/4*E1  + 1/4*Δt*dE
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx = 1/3*μx + 2/3*μx2 + 2/3*Δt*dμx
        @. μy = 1/3*μy + 2/3*μy2 + 2/3*Δt*dμy
        @. ρ  = 1/3*ρ  + 2/3*ρ2  + 2/3*Δt*dρ
        @. E  = 1/3*E  + 2/3*E2  + 2/3*Δt*dE
    end

    J_val = zero(eltype(ρ))
    for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
        g = sin(2π*mesh.x[i,j,ex,ey]/mesh.Lx) * sin(2π*mesh.y[i,j,ex,ey]/mesh.Ly)
        J_val += basis.w[i] * basis.w[j] * mesh.Jx * mesh.Jy * μx[i,j,ex,ey] * g
    end
    return J_val
end

# ============================================================
# Objective dispatch
# ============================================================

"""
    enzyme_objective_fn(objective::Symbol)

Return the top-level objective function for the given symbol.
"""
function enzyme_objective_fn(objective::Symbol)
    if objective == :l2_density
        return _enzyme_objective
    elseif objective == :l2_pressure
        return _enzyme_objective_l2_pressure
    elseif objective == :kinetic_energy
        return _enzyme_objective_kinetic_energy
    elseif objective == :windowed_kinetic_energy
        return _enzyme_objective_windowed_kinetic_energy
    elseif objective == :weighted_momentum
        return _enzyme_objective_weighted_momentum
    else
        error("Unknown objective: $objective")
    end
end

# ============================================================
# Gradient driver
# ============================================================

"""
    enzyme_adjoint_ic(μx0, μy0, ρ0, E0, n_steps, Δt, basis, mesh, γ, α;
                      n_iter=10, solver=:pcg, objective=:l2_density)

Compute the gradient of the scalar objective with respect to the
initial condition via Enzyme reverse mode.

Returns `(J, dμx0, dμy0, dρ0, dE0)` — the objective value and the
conservative-variable gradients `∂J/∂q₀`, each `(n_p,n_p,N_ex,N_ey)`.
"""
function enzyme_adjoint_ic(μx0, μy0, ρ0, E0, n_steps, Δt, basis, mesh, γ, α;
                           n_iter::Int=10, solver::Symbol=:pcg, objective::Symbol=:l2_density)
    n_p  = basis.p + 1
    N_ex = mesh.N_ex
    N_ey = mesh.N_ey
    obj_fn = enzyme_objective_fn(objective)

    dμx0 = zeros(n_p, n_p, N_ex, N_ey)
    dμy0 = zeros(n_p, n_p, N_ex, N_ey)
    dρ0  = zeros(n_p, n_p, N_ex, N_ey)
    dE0  = zeros(n_p, n_p, N_ex, N_ey)

    mode = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
    result = Enzyme.autodiff(mode, obj_fn, Enzyme.Active,
        Enzyme.Const(α),
        Enzyme.Duplicated(copy(μx0), dμx0),
        Enzyme.Duplicated(copy(μy0), dμy0),
        Enzyme.Duplicated(copy(ρ0),  dρ0),
        Enzyme.Duplicated(copy(E0),  dE0),
        Enzyme.Const(n_p), Enzyme.Const(N_ex), Enzyme.Const(N_ey),
        Enzyme.Const(n_steps), Enzyme.Const(Δt),
        Enzyme.Const(basis), Enzyme.Const(mesh),
        Enzyme.Const(γ), Enzyme.Const(max(n_iter, 1)), Enzyme.Const(solver))

    J_val = result[2]
    return J_val, dμx0, dμy0, dρ0, dE0
end
