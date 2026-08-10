# ============================================================
# Enzyme AD wrappers for the IGR Euler solver
#
# Uses Enzyme reverse mode for efficient scalar-objective gradients.
#
# Key workarounds for Enzyme v0.13 + Julia 1.12:
#   1. Inlined SSP-RK3 loop: calling run_forward!() triggers an
#      opaque-pointer bug in Enzyme's cache mechanism at the
#      augmented forward/reverse pass boundary. The SSP-RK3 loop
#      is therefore inlined in each objective function. The loop
#      body is identical to run_forward!() in forward.jl.
#   2. Top-level objectives: Enzyme needs top-level functions (not closures)
#      with explicit Const/Active annotations for each argument.
#   3. set_runtime_activity for struct field access (DGBasis, PeriodicMesh1D)
#   4. No keyword arguments in the differentiated call chain
# ============================================================

export enzyme_dJ_dA, enzyme_adjoint_ic

# ============================================================
# Top-level objective function for Enzyme reverse mode
#
# This is NOT a closure — all arguments are explicit so Enzyme
# can annotate each one as Active or Const.
# ============================================================

"""
    _enzyme_objective(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)

Top-level objective J = ∫ρ(x,T)² dx for Enzyme reverse-mode differentiation.
SSP-RK3 is inlined (not calling run_forward!) to avoid Enzyme's opaque-pointer bug.
All arguments are explicit (no closures) for Enzyme compatibility.
"""
function _enzyme_objective(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)
    # State arrays
    μ = copy(μ0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, N_e)

    # Stage arrays
    μ1 = similar(μ); ρ1 = similar(ρ); E1 = similar(E)
    μ2 = similar(μ); ρ2 = similar(ρ); E2 = similar(E)
    dμ = similar(μ); dρ = similar(ρ); dE = similar(E)

    # SSP-RK3 time integration (inlined for Enzyme compatibility)
    for _ in 1:n_steps
        # Stage 1
        compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ1 = μ + Δt * dμ
        @. ρ1 = ρ + Δt * dρ
        @. E1 = E + Δt * dE

        # Stage 2
        compute_igr_euler_rhs!(dμ, dρ, dE, μ1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ2 = 3/4 * μ + 1/4 * μ1 + 1/4 * Δt * dμ
        @. ρ2 = 3/4 * ρ + 1/4 * ρ1 + 1/4 * Δt * dρ
        @. E2 = 3/4 * E + 1/4 * E1 + 1/4 * Δt * dE

        # Stage 3
        compute_igr_euler_rhs!(dμ, dρ, dE, μ2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ = 1/3 * μ + 2/3 * μ2 + 2/3 * Δt * dμ
        @. ρ = 1/3 * ρ + 2/3 * ρ2 + 2/3 * Δt * dρ
        @. E = 1/3 * E + 2/3 * E2 + 2/3 * Δt * dE
    end

    J_val = zero(eltype(ρ))
    for e in 1:N_e
        for i in 1:n_p
            J_val += basis.w[i] * mesh.J * ρ[i,e]^2
        end
    end
    return J_val
end

"""
    _enzyme_objective_l2_pressure(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)

Top-level objective J = ∫p(x,T)² dx for Enzyme reverse-mode differentiation.
"""
function _enzyme_objective_l2_pressure(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)
    # State arrays
    μ = copy(μ0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, N_e)

    # Stage arrays
    μ1 = similar(μ); ρ1 = similar(ρ); E1 = similar(E)
    μ2 = similar(μ); ρ2 = similar(ρ); E2 = similar(E)
    dμ = similar(μ); dρ = similar(ρ); dE = similar(E)

    # SSP-RK3 time integration (inlined for Enzyme compatibility)
    for _ in 1:n_steps
        # Stage 1
        compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ1 = μ + Δt * dμ
        @. ρ1 = ρ + Δt * dρ
        @. E1 = E + Δt * dE

        # Stage 2
        compute_igr_euler_rhs!(dμ, dρ, dE, μ1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ2 = 3/4 * μ + 1/4 * μ1 + 1/4 * Δt * dμ
        @. ρ2 = 3/4 * ρ + 1/4 * ρ1 + 1/4 * Δt * dρ
        @. E2 = 3/4 * E + 1/4 * E1 + 1/4 * Δt * dE

        # Stage 3
        compute_igr_euler_rhs!(dμ, dρ, dE, μ2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ = 1/3 * μ + 2/3 * μ2 + 2/3 * Δt * dμ
        @. ρ = 1/3 * ρ + 2/3 * ρ2 + 2/3 * Δt * dρ
        @. E = 1/3 * E + 2/3 * E2 + 2/3 * Δt * dE
    end

    J_val = zero(eltype(ρ))
    for e in 1:N_e
        for i in 1:n_p
            p_i = (γ - 1) * (E[i,e] - μ[i,e]^2 / (2 * ρ[i,e]))
            J_val += basis.w[i] * mesh.J * p_i^2
        end
    end
    return J_val
end

"""
    _enzyme_objective_weighted_momentum(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)

Top-level objective J = ∫ μ(x,T) sin(2πx/L) dx for Enzyme reverse-mode differentiation.
Sine-weighted momentum at final time — smooth, linear functional.
"""
function _enzyme_objective_weighted_momentum(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)
    # State arrays
    μ = copy(μ0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, N_e)

    # Stage arrays
    μ1 = similar(μ); ρ1 = similar(ρ); E1 = similar(E)
    μ2 = similar(μ); ρ2 = similar(ρ); E2 = similar(E)
    dμ = similar(μ); dρ = similar(ρ); dE = similar(E)

    # SSP-RK3 time integration (inlined for Enzyme compatibility)
    for _ in 1:n_steps
        # Stage 1
        compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ1 = μ + Δt * dμ
        @. ρ1 = ρ + Δt * dρ
        @. E1 = E + Δt * dE

        # Stage 2
        compute_igr_euler_rhs!(dμ, dρ, dE, μ1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ2 = 3/4 * μ + 1/4 * μ1 + 1/4 * Δt * dμ
        @. ρ2 = 3/4 * ρ + 1/4 * ρ1 + 1/4 * Δt * dρ
        @. E2 = 3/4 * E + 1/4 * E1 + 1/4 * Δt * dE

        # Stage 3
        compute_igr_euler_rhs!(dμ, dρ, dE, μ2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ = 1/3 * μ + 2/3 * μ2 + 2/3 * Δt * dμ
        @. ρ = 1/3 * ρ + 2/3 * ρ2 + 2/3 * Δt * dρ
        @. E = 1/3 * E + 2/3 * E2 + 2/3 * Δt * dE
    end

    L_domain = mesh.N_e * mesh.Δx
    J_val = zero(eltype(ρ))
    for e in 1:N_e
        for i in 1:n_p
            J_val += basis.w[i] * mesh.J * μ[i,e] * sin(2π * mesh.x[i,e] / L_domain)
        end
    end
    return J_val
end

"""
    _enzyme_objective_kinetic_energy(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)

Top-level objective J = (∫ μ²/(2ρ) dx) / (∫ ρ dx) for Enzyme reverse-mode differentiation.
This is the mass-averaged kinetic energy at final time T.
"""
function _enzyme_objective_kinetic_energy(α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt, basis, mesh, γ, n_iter, solver)
    # State arrays
    μ = copy(μ0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(n_p, N_e)

    # Stage arrays
    μ1 = similar(μ); ρ1 = similar(ρ); E1 = similar(E)
    μ2 = similar(μ); ρ2 = similar(ρ); E2 = similar(E)
    dμ = similar(μ); dρ = similar(ρ); dE = similar(E)

    # SSP-RK3 time integration (inlined for Enzyme compatibility)
    for _ in 1:n_steps
        # Stage 1
        compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ1 = μ + Δt * dμ
        @. ρ1 = ρ + Δt * dρ
        @. E1 = E + Δt * dE

        # Stage 2
        compute_igr_euler_rhs!(dμ, dρ, dE, μ1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ2 = 3/4 * μ + 1/4 * μ1 + 1/4 * Δt * dμ
        @. ρ2 = 3/4 * ρ + 1/4 * ρ1 + 1/4 * Δt * dρ
        @. E2 = 3/4 * E + 1/4 * E1 + 1/4 * Δt * dE

        # Stage 3
        compute_igr_euler_rhs!(dμ, dρ, dE, μ2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μ = 1/3 * μ + 2/3 * μ2 + 2/3 * Δt * dμ
        @. ρ = 1/3 * ρ + 2/3 * ρ2 + 2/3 * Δt * dρ
        @. E = 1/3 * E + 2/3 * E2 + 2/3 * Δt * dE
    end

    # J = KE / M = (∫ μ²/(2ρ) dx) / (∫ ρ dx)
    KE = zero(eltype(ρ))
    M = zero(eltype(ρ))
    for e in 1:N_e
        for i in 1:n_p
            wJ = basis.w[i] * mesh.J
            KE += wJ * μ[i,e]^2 / (2 * ρ[i,e])
            M  += wJ * ρ[i,e]
        end
    end
    return KE / M
end

# ============================================================
# User-facing gradient drivers
# ============================================================

"""
    enzyme_dJ_dA(A0, p, N_e, L, γ, α, CFL, T; n_iter=10, ic_type=:acoustic_pulse)

Compute dJ/dA at A=A₀ using Enzyme reverse-mode AD.
J = ∫ρ(x,T)² dx where the IC is parameterized by amplitude A.

Uses `enzyme_adjoint_ic` to get the full adjoint fields, then projects
onto ∂IC/∂A via the chain rule: dJ/dA = ∫ (wρ·adj_ρ + wu·adj_u + wp·adj_p)·sin(2πx/L) dx.

Returns (J, dJ_dA).
"""
function enzyme_dJ_dA(A0, p, N_e, L, γ, α, CFL, T; n_iter::Int=10, ic_type::Symbol=:acoustic_pulse, solver::Symbol=:pcg)
    J_val, adj_ρ, adj_u, adj_p = enzyme_adjoint_ic(A0, p, N_e, L, γ, α, CFL, T;
                                                     n_iter=n_iter, ic_type=ic_type, solver=solver)

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1
    wρ, wu, wp = ic_dA_weights(ic_type)

    dJ_dA = 0.0
    for e in 1:N_e, i in 1:n_p
        s = sin(2π * mesh.x[i,e] / L)
        dJ_dA += (wρ * adj_ρ[i,e] + wu * adj_u[i,e] + wp * adj_p[i,e]) * s
    end

    return J_val, dJ_dA
end

"""
    enzyme_adjoint_ic(A, p, N_e, L, γ, α, CFL, T; n_iter=10, ic_type=:acoustic_pulse)

Compute the adjoint of J = ∫ρ(x,T)² dx with respect to the initial condition,
returning the gradient in primitive variables: ∂J/∂ρ₀(x), ∂J/∂u₀(x), ∂J/∂p₀(x).

Uses Enzyme reverse mode with `Duplicated` IC arrays to obtain ∂J/∂(μ₀, ρ₀, E₀),
then transforms from conservative to primitive via the transpose Jacobian:

    ∂J/∂ρ_prim = ∂J/∂ρ + u·∂J/∂μ + (u²/2)·∂J/∂E
    ∂J/∂u      = ρ·∂J/∂μ + ρu·∂J/∂E
    ∂J/∂p      = (1/(γ-1))·∂J/∂E

The `ic_type` keyword selects the initial condition family (see `IC_BUILDERS`).

Returns (J, adj_ρ, adj_u, adj_p) where each adj field is (n_p × N_e).
"""
function enzyme_adjoint_ic(A, p, N_e, L, γ, α, CFL, T; n_iter::Int=10, ic_type::Symbol=:acoustic_pulse, solver::Symbol=:pcg, objective::Symbol=:l2_density)
    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)
    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    # Build IC using selected type
    μ0 = zeros(n_p, N_e); ρ0 = zeros(n_p, N_e); E0 = zeros(n_p, N_e)
    builder! = IC_BUILDERS[ic_type][1]
    builder!(μ0, ρ0, E0, A, mesh.x, L, γ)

    # Shadow arrays for Enzyme (will be filled with ∂J/∂q₀)
    dμ0 = zeros(n_p, N_e); dρ0 = zeros(n_p, N_e); dE0 = zeros(n_p, N_e)

    # Select objective function
    obj_fn = if objective == :l2_density
        _enzyme_objective
    elseif objective == :l2_pressure
        _enzyme_objective_l2_pressure
    else
        error("Unknown objective: $objective (use :l2_density or :l2_pressure)")
    end

    mode = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
    result = Enzyme.autodiff(mode, obj_fn, Enzyme.Active,
        Enzyme.Const(α),
        Enzyme.Duplicated(μ0, dμ0),
        Enzyme.Duplicated(ρ0, dρ0),
        Enzyme.Duplicated(E0, dE0),
        Enzyme.Const(n_p), Enzyme.Const(N_e),
        Enzyme.Const(n_steps), Enzyme.Const(Δt),
        Enzyme.Const(basis), Enzyme.Const(mesh),
        Enzyme.Const(γ), Enzyme.Const(max(n_iter, 1)), Enzyme.Const(solver))

    J_val = result[2]

    # Transform conservative adjoints → primitive adjoints
    u0 = μ0 ./ ρ0
    adj_ρ = dρ0 .+ u0 .* dμ0 .+ (u0.^2 ./ 2) .* dE0
    adj_u = ρ0 .* dμ0 .+ ρ0 .* u0 .* dE0
    adj_p = dE0 ./ (γ - 1)

    return J_val, adj_ρ, adj_u, adj_p
end

