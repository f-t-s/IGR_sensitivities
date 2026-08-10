# ============================================================
# Combined IGR Euler forward solver: elliptic + hyperbolic RHS
# ============================================================

export compute_igr_euler_rhs!, run_forward!, run_forward, run_forward_store

"""
    compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α;
                           n_iter=10)

Compute the full IGR Euler right-hand side:
1. Solve the elliptic equation for Σ (if α > 0)
2. Evaluate the hyperbolic DG operator with the computed Σ

This is the spatial operator R_h(q, Σ(q)) from the paper.
"""
function compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α;
                                n_iter=10, solver=:pcg)
    return compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
end

# Positional-argument form (avoids kwcall for Enzyme compatibility on Julia 1.12)
function compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α, n_iter::Int, solver::Symbol=:pcg)
    if α > 0
        solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α, n_iter, solver)
    else
        fill!(Σ, zero(eltype(Σ)))
    end

    compute_hyperbolic_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ)

    return nothing
end

"""
    run_forward!(μ, ρ, E, Σ, Δt, T, basis, mesh, γ, α, n_iter::Int, solver::Symbol=:pcg)

In-place SSP-RK3 forward solve. On entry, `μ, ρ, E` must contain the
initial condition; on exit they contain the solution at time T.
`Σ` is workspace for the elliptic variable.

This is the single forward pass implementation used by all code paths:
direct evaluation, ForwardDiff (via `run_forward`), Enzyme reverse mode
(via `_enzyme_objective`), and the adjoint PDE solver (via `run_forward_store`).

Returns `nothing` — all output is written into the mutable arguments.
The mutating interface avoids returning a tuple of arrays, which
triggers an opaque-pointer bug in Enzyme v0.13.
"""
function run_forward!(μ, ρ, E, Σ, Δt, T, basis, mesh, γ, α, n_iter::Int, solver::Symbol=:pcg)
    fill!(Σ, zero(eltype(Σ)))

    # Stage arrays
    μ1 = similar(μ); ρ1 = similar(ρ); E1 = similar(E)
    μ2 = similar(μ); ρ2 = similar(ρ); E2 = similar(E)
    dμ = similar(μ); dρ = similar(ρ); dE = similar(E)

    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    for step in 1:n_steps
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

    # Compute Σ at final state
    if α > 0
        solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α, n_iter, solver)
    end

    return nothing
end

"""
    run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α, n_iter::Int, solver::Symbol=:pcg)

Allocating wrapper around `run_forward!`. Copies the initial condition,
runs the forward solve, and returns (μ, ρ, E, Σ) at final time.

Positional-argument form for AD compatibility.
"""
function run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α, n_iter::Int, solver::Symbol=:pcg)
    μ = copy(μ0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(eltype(μ), size(μ))
    run_forward!(μ, ρ, E, Σ, Δt, T, basis, mesh, γ, α, n_iter, solver)
    return μ, ρ, E, Σ
end

"""
    run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=10, solver=:pcg)

Keyword-argument convenience form with type promotion for ForwardDiff.
Promotes all numeric arrays to a common type so Dual numbers propagate correctly.
"""
function run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α;
                     n_iter=10, solver=:pcg)
    # Promote to common numeric type for AD compatibility (no-op for Float64)
    RT = promote_type(eltype(μ0), eltype(ρ0), eltype(E0),
                      typeof(α), typeof(γ), typeof(Δt), typeof(T))
    if RT !== eltype(μ0); μ0 = RT.(μ0); end
    if RT !== eltype(ρ0); ρ0 = RT.(ρ0); end
    if RT !== eltype(E0); E0 = RT.(E0); end

    return run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α, n_iter, solver)
end

"""
    run_forward_store(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=10)

Run the forward solve and store snapshots of (μ, ρ, E, Σ) at each
time step for use by the adjoint solver.

Uses the same SSP-RK3 integration and RHS evaluation as `run_forward`,
but additionally stores (μ, ρ, E, Σ) at each time step. This variant
is NOT differentiated through — it only provides snapshots for the
continuous adjoint PDE solver.

Returns (snapshots, n_steps) where snapshots is a vector of
(μ, ρ, E, Σ) tuples, indexed from 0 to n_steps.
snapshots[1] = state at t=0, snapshots[end] = state at t=T.
"""
function run_forward_store(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=10, solver=:pcg)
    RT = promote_type(eltype(μ0), eltype(ρ0), eltype(E0),
                      typeof(α), typeof(γ), typeof(Δt), typeof(T))
    μ = RT.(copy(μ0)); ρ = RT.(copy(ρ0)); E = RT.(copy(E0))
    Σ = zeros(RT, size(μ))

    # Allocate stage arrays
    μ1 = similar(μ); ρ1 = similar(ρ); E1 = similar(E)
    μ2 = similar(μ); ρ2 = similar(ρ); E2 = similar(E)
    dμ = similar(μ); dρ = similar(ρ); dE = similar(E)

    # Store initial state
    snapshots = Tuple{Matrix{RT}, Matrix{RT}, Matrix{RT}, Matrix{RT}}[]

    # Compute Σ at initial state
    if α > 0
        solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α; n_iter=n_iter, solver=solver)
    end
    push!(snapshots, (copy(μ), copy(ρ), copy(E), copy(Σ)))

    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    for step in 1:n_steps
        # Stage 1
        compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μ1 = μ + Δt * dμ
        @. ρ1 = ρ + Δt * dρ
        @. E1 = E + Δt * dE

        # Stage 2
        compute_igr_euler_rhs!(dμ, dρ, dE, μ1, ρ1, E1, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μ2 = 3/4 * μ + 1/4 * μ1 + 1/4 * Δt * dμ
        @. ρ2 = 3/4 * ρ + 1/4 * ρ1 + 1/4 * Δt * dρ
        @. E2 = 3/4 * E + 1/4 * E1 + 1/4 * Δt * dE

        # Stage 3
        compute_igr_euler_rhs!(dμ, dρ, dE, μ2, ρ2, E2, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μ = 1/3 * μ + 2/3 * μ2 + 2/3 * Δt * dμ
        @. ρ = 1/3 * ρ + 2/3 * ρ2 + 2/3 * Δt * dρ
        @. E = 1/3 * E + 2/3 * E2 + 2/3 * Δt * dE

        # Compute and store Σ at new state
        if α > 0
            solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α; n_iter=n_iter, solver=solver)
        end
        push!(snapshots, (copy(μ), copy(ρ), copy(E), copy(Σ)))
    end

    return snapshots, n_steps
end