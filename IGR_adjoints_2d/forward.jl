# ============================================================
# Combined 2D IGR Euler forward solver: elliptic + hyperbolic RHS,
# SSP-RK3 time integration.
# ============================================================

export compute_igr_euler_rhs!, run_forward!, run_forward, run_forward_store

"""
    compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)

Full 2D IGR Euler spatial operator:
1. Solve the elliptic equation for Σ (if α > 0)
2. Evaluate the hyperbolic DG operator with the computed Σ.
"""
function compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α;
                                n_iter=10, solver=:pcg)
    return compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α,
                                  n_iter, solver)
end

# Positional-argument form (avoids kwcall for Enzyme compatibility)
function compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α,
                                n_iter::Int, solver::Symbol=:pcg)
    if α > 0
        solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α, n_iter, solver)
    else
        fill!(Σ, zero(eltype(Σ)))
    end
    compute_hyperbolic_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ)
    return nothing
end

"""
    run_forward!(μx, μy, ρ, E, Σ, Δt, T, basis, mesh, γ, α, n_iter, solver)

In-place SSP-RK3 forward solve. On entry the state arrays hold the
initial condition; on exit they hold the solution at time T. `Σ` is
workspace for the elliptic variable. All output is written into the
mutable arguments (mutating interface avoids an Enzyme opaque-pointer bug).
"""
function run_forward!(μx, μy, ρ, E, Σ, Δt, T, basis, mesh, γ, α, n_iter::Int, solver::Symbol=:pcg)
    fill!(Σ, zero(eltype(Σ)))

    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)

    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    for _ in 1:n_steps
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx1 = μx + Δt * dμx
        @. μy1 = μy + Δt * dμy
        @. ρ1  = ρ  + Δt * dρ
        @. E1  = E  + Δt * dE

        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx2 = 3/4 * μx + 1/4 * μx1 + 1/4 * Δt * dμx
        @. μy2 = 3/4 * μy + 1/4 * μy1 + 1/4 * Δt * dμy
        @. ρ2  = 3/4 * ρ  + 1/4 * ρ1  + 1/4 * Δt * dρ
        @. E2  = 3/4 * E  + 1/4 * E1  + 1/4 * Δt * dE

        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α, n_iter, solver)
        @. μx = 1/3 * μx + 2/3 * μx2 + 2/3 * Δt * dμx
        @. μy = 1/3 * μy + 2/3 * μy2 + 2/3 * Δt * dμy
        @. ρ  = 1/3 * ρ  + 2/3 * ρ2  + 2/3 * Δt * dρ
        @. E  = 1/3 * E  + 2/3 * E2  + 2/3 * Δt * dE
    end

    if α > 0
        solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α, n_iter, solver)
    end
    return nothing
end

"""
    run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=10, solver=:pcg)

Allocating wrapper around `run_forward!`. Promotes the initial condition
to a common numeric type (for AD), runs the solve, and returns
`(μx, μy, ρ, E, Σ)` at the final time.
"""
function run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α, n_iter::Int, solver::Symbol=:pcg)
    μx = copy(μx0); μy = copy(μy0); ρ = copy(ρ0); E = copy(E0)
    Σ = zeros(eltype(μx), size(μx))
    run_forward!(μx, μy, ρ, E, Σ, Δt, T, basis, mesh, γ, α, n_iter, solver)
    return μx, μy, ρ, E, Σ
end

function run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=10, solver=:pcg)
    RT = promote_type(eltype(μx0), eltype(μy0), eltype(ρ0), eltype(E0),
                      typeof(α), typeof(γ), typeof(Δt), typeof(T))
    μx0 = RT === eltype(μx0) ? μx0 : RT.(μx0)
    μy0 = RT === eltype(μy0) ? μy0 : RT.(μy0)
    ρ0  = RT === eltype(ρ0)  ? ρ0  : RT.(ρ0)
    E0  = RT === eltype(E0)  ? E0  : RT.(E0)
    return run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α, n_iter, solver)
end

"""
    run_forward_store(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=10, solver=:pcg)

Run the forward solve and store snapshots of `(μx, μy, ρ, E, Σ)` at every
time step for use by the adjoint solver. Same SSP-RK3 integration as
`run_forward!`; this variant is not differentiated through.

Returns `(snapshots, n_steps)` with `snapshots[1]` the state at t=0 and
`snapshots[end]` the state at t=T.
"""
function run_forward_store(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=10, solver=:pcg)
    RT = promote_type(eltype(μx0), eltype(μy0), eltype(ρ0), eltype(E0),
                      typeof(α), typeof(γ), typeof(Δt), typeof(T))
    μx = RT.(copy(μx0)); μy = RT.(copy(μy0)); ρ = RT.(copy(ρ0)); E = RT.(copy(E0))
    Σ = zeros(RT, size(μx))

    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)

    A4 = Array{RT,4}
    snapshots = Tuple{A4,A4,A4,A4,A4}[]

    if α > 0
        solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α; n_iter=n_iter, solver=solver)
    end
    push!(snapshots, (copy(μx), copy(μy), copy(ρ), copy(E), copy(Σ)))

    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    for _ in 1:n_steps
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μx1 = μx + Δt * dμx
        @. μy1 = μy + Δt * dμy
        @. ρ1  = ρ  + Δt * dρ
        @. E1  = E  + Δt * dE

        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μx2 = 3/4 * μx + 1/4 * μx1 + 1/4 * Δt * dμx
        @. μy2 = 3/4 * μy + 1/4 * μy1 + 1/4 * Δt * dμy
        @. ρ2  = 3/4 * ρ  + 1/4 * ρ1  + 1/4 * Δt * dρ
        @. E2  = 3/4 * E  + 1/4 * E1  + 1/4 * Δt * dE

        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μx = 1/3 * μx + 2/3 * μx2 + 2/3 * Δt * dμx
        @. μy = 1/3 * μy + 2/3 * μy2 + 2/3 * Δt * dμy
        @. ρ  = 1/3 * ρ  + 2/3 * ρ2  + 2/3 * Δt * dρ
        @. E  = 1/3 * E  + 2/3 * E2  + 2/3 * Δt * dE

        if α > 0
            solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α; n_iter=n_iter, solver=solver)
        end
        push!(snapshots, (copy(μx), copy(μy), copy(ρ), copy(E), copy(Σ)))
    end

    return snapshots, n_steps
end
