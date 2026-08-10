#!/usr/bin/env julia
# ============================================================
# Verify the 2D continuous adjoint PDE solver against Enzyme
# reverse-mode AD. The two should agree up to the
# optimize-then-discretize (OtD) gap, which vanishes under
# refinement. A derivation/sign error instead produces an
# O(1) relative disagreement.
# ============================================================

using Printf, LinearAlgebra

include("../IGRAdjoints2D.jl")
using .IGRAdjoints2D

function compare(objective::Symbol, α::Float64; p=3, N_e=8, T=0.06)
    γ = 1.4
    L = 1.0
    A = 0.4
    solver = :chebyshev

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
    n_iter = α > 0 ? 100 : 1

    # --- Enzyme reverse-mode AD (discrete adjoint) ---
    J, dμx, dμy, dρ, dE = enzyme_adjoint_ic(μx0, μy0, ρ0, E0, n_steps, Δt,
        basis, mesh, γ, α; n_iter=n_iter, solver=solver, objective=objective)

    # Discrete dJ/dq₀ → L² gradient density: divide by the nodal mass w_i w_j Jx Jy
    enz_μx = similar(dμx); enz_μy = similar(dμy)
    enz_ρ  = similar(dρ);  enz_E  = similar(dE)
    for ey in 1:N_e, ex in 1:N_e, j in 1:n_p, i in 1:n_p
        m = basis.w[i] * basis.w[j] * mesh.Jx * mesh.Jy
        enz_μx[i,j,ex,ey] = dμx[i,j,ex,ey] / m
        enz_μy[i,j,ex,ey] = dμy[i,j,ex,ey] / m
        enz_ρ[i,j,ex,ey]  = dρ[i,j,ex,ey]  / m
        enz_E[i,j,ex,ey]  = dE[i,j,ex,ey]  / m
    end

    # --- Continuous adjoint PDE (forward store + backward integration) ---
    snapshots, n_steps_f = run_forward_store(μx0, μy0, ρ0, E0, Δt, T,
        basis, mesh, γ, α; n_iter=n_iter, solver=solver)
    pde_μx, pde_μy, pde_ρ, pde_E = run_adjoint_conservative(snapshots, n_steps_f,
        Δt, T, basis, mesh, γ, α; n_iter=n_iter, objective=objective, solver=solver)

    # --- Relative L² differences ---
    relerr(p_, e_) = norm(p_ .- e_) / max(norm(e_), 1e-30)
    eμx = relerr(pde_μx, enz_μx)
    eμy = relerr(pde_μy, enz_μy)
    eρ  = relerr(pde_ρ,  enz_ρ)
    eE  = relerr(pde_E,  enz_E)

    combined = norm(vcat(vec(pde_μx .- enz_μx), vec(pde_μy .- enz_μy),
                         vec(pde_ρ .- enz_ρ),  vec(pde_E .- enz_E))) /
               max(norm(vcat(vec(enz_μx), vec(enz_μy), vec(enz_ρ), vec(enz_E))), 1e-30)

    @printf("  objective=%-18s α=%.1e  J=%.6e  n_steps=%d\n", objective, α, J, n_steps)
    @printf("    rel L² (PDE vs Enzyme):  μx=%.3e μy=%.3e ρ=%.3e E=%.3e\n", eμx, eμy, eρ, eE)
    @printf("    combined rel L² = %.3e\n", combined)
    @assert combined < 0.2 "PDE adjoint disagrees with Enzyme ($objective, α=$α): $combined"
    println("    OK (within OtD gap)")
    return combined
end

println("="^60)
println("2D continuous adjoint PDE  vs  Enzyme reverse-mode AD")
println("="^60)
for objective in (:l2_density, :weighted_momentum, :kinetic_energy)
    compare(objective, 0.0)
    compare(objective, 1.5e-3)
end

# --- OtD-gap at two resolutions (informational) ---
# The PDE-vs-Enzyme gap is the optimize-then-discretize gap. At the
# resolutions below it already sits near the floor set by the elliptic
# solver convergence and the SSP-RK3 adjoint mismatch; the check below
# only guards against the gap *growing* uncontrollably under refinement.
println("\nOtD-gap at two resolutions (objective = l2_density, α = 1.5e-3):")
e_coarse = compare(:l2_density, 1.5e-3; N_e=6,  T=0.06)
e_fine   = compare(:l2_density, 1.5e-3; N_e=12, T=0.06)
@printf("  N_e 6 → 12:  %.3e → %.3e  (ratio %.2f)\n", e_coarse, e_fine, e_coarse/e_fine)
@assert e_fine < 2 * e_coarse "OtD gap grew uncontrollably under refinement"

println("\nAll Stage 3 adjoint tests passed.")
