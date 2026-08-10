#!/usr/bin/env julia
# ============================================================
# Validate the 2D Enzyme reverse-mode gradient against central
# finite differences of the same objective.
# ============================================================

using Printf, Random

include("../IGRAdjoints2D.jl")
using .IGRAdjoints2D

const M = IGRAdjoints2D  # access internal objective functions

function run_test(objective::Symbol, α::Float64)
    γ = 1.4
    L = 1.0
    p = 2
    N_e = 4
    A = 0.4
    T = 0.08

    basis = DGBasis(p)
    mesh  = CartesianMesh2D(N_e, N_e, L, L, basis)
    n_p = p + 1

    μx0 = similar(mesh.x); μy0 = similar(mesh.x)
    ρ0  = similar(mesh.x); E0  = similar(mesh.x)
    build_taylor_green_ic!(μx0, μy0, ρ0, E0, A, mesh, γ)

    max_ws = 0.0
    for idx in eachindex(μx0)
        ws = max(max_wavespeed_x(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]),
                 max_wavespeed_y(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]))
        max_ws = max(max_ws, ws)
    end
    Δt = 0.4 * min(mesh.Δx, mesh.Δy) / ((2*p + 1) * max_ws)
    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps
    n_iter = α > 0 ? 60 : 1
    solver = :chebyshev

    obj_fn = M.enzyme_objective_fn(objective)

    # Enzyme reverse-mode gradient
    J, dμx, dμy, dρ, dE = enzyme_adjoint_ic(μx0, μy0, ρ0, E0, n_steps, Δt,
        basis, mesh, γ, α; n_iter=n_iter, solver=solver, objective=objective)

    # Finite-difference gradient at a few probe indices
    Random.seed!(42)
    idxs = rand(1:length(μx0), 6)
    h = 1e-6
    obj(mx, my, r, e) = obj_fn(α, mx, my, r, e, n_p, N_e, N_e, n_steps, Δt,
                               basis, mesh, γ, n_iter, solver)

    function fd_field(field, grad_arr)
        errs = Float64[]
        for idx in idxs
            base = copy(field)
            fp = copy(base); fp[idx] += h
            fm = copy(base); fm[idx] -= h
            if field === μx0
                Jp = obj(fp, μy0, ρ0, E0); Jm = obj(fm, μy0, ρ0, E0)
            elseif field === μy0
                Jp = obj(μx0, fp, ρ0, E0); Jm = obj(μx0, fm, ρ0, E0)
            elseif field === ρ0
                Jp = obj(μx0, μy0, fp, E0); Jm = obj(μx0, μy0, fm, E0)
            else
                Jp = obj(μx0, μy0, ρ0, fp); Jm = obj(μx0, μy0, ρ0, fm)
            end
            g_fd = (Jp - Jm) / (2h)
            push!(errs, abs(g_fd - grad_arr[idx]))
        end
        return maximum(errs), maximum(abs.(grad_arr))
    end

    eμx, sμx = fd_field(μx0, dμx)
    eμy, sμy = fd_field(μy0, dμy)
    eρ,  sρ  = fd_field(ρ0,  dρ)
    eE,  sE  = fd_field(E0,  dE)

    scale = max(sμx, sμy, sρ, sE, 1e-30)
    relerr = max(eμx, eμy, eρ, eE) / scale
    @printf("  objective=%-18s α=%.1e  J=%.6e  n_steps=%d\n", objective, α, J, n_steps)
    @printf("    max|FD-Enzyme|:  μx=%.2e μy=%.2e ρ=%.2e E=%.2e   rel=%.2e\n",
            eμx, eμy, eρ, eE, relerr)
    # Tolerance reflects central-difference accuracy (truncation O(h²) plus
    # roundoff O(ε/h)); a genuine bug yields O(1) relative disagreement.
    @assert relerr < 1e-3 "Enzyme gradient disagrees with FD ($objective, α=$α): rel=$relerr"
    println("    PASSED")
    return nothing
end

println("="^60)
println("2D Enzyme reverse-mode gradient vs finite differences")
println("="^60)
for objective in (:l2_density, :weighted_momentum, :kinetic_energy)
    run_test(objective, 0.0)
    run_test(objective, 1.5e-3)
end
println("\nAll Stage 2 Enzyme tests passed.")
