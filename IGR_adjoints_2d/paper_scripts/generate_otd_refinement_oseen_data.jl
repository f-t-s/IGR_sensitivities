#!/usr/bin/env julia
# ============================================================
# Generate CSV data for the optimize-then-discretize (OtD) gap
# refinement study on the Oseen-pair / blast configuration.
#
# A Gaussian over-pressure blast sweeps over a counter-rotating
# pair of physical Oseen vortices (Colonius, Lele & Moin, JFM
# 1991 initialization; see build_oseen_pair_blast_ic!). The
# objective is the kinetic energy in a Gaussian window over the
# vortex-wake region at t = T (:windowed_kinetic_energy, window
# KE_WINDOW), and we compare
#   - the continuous-adjoint directional derivative dJ/dA
#     (A = blast amplitude) with
#   - a central finite difference of the discrete forward map,
# across mesh refinement. Companion of the three-Sedov-blast
# study in generate_otd_refinement_sedov_data.jl.
#
# Output (in figures/data/):
#   otd_refinement_oseen.csv   — columns h, gap, adjoint, fd
# ============================================================

using Printf, LinearAlgebra

include(joinpath(@__DIR__, "..", "IGRAdjoints2D.jl"))
using .IGRAdjoints2D

# Figure data is written here, inside this repository. Copy the contents of
# paper_data/ into the paper's figures/data/ directory to rebuild the figures.
const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper_data")
mkpath(DATA_DIR)

# ============================================================
# Fixed settings (identical for every resolution; the Oseen-pair
# and blast geometry are the build_oseen_pair_blast_ic! defaults)
# ============================================================
γ   = 1.4
L   = 1.0
p   = 4
T   = 0.10
CFL = 0.4
α   = 3.0 * (L / 20)^2     # FIXED physical IGR regularization (no mesh dependence)
n_iter = 60                # PCG iterations — enough to converge the elliptic solves
solver = :pcg
objective = :windowed_kinetic_energy

blast_A = 12.0             # blast over-pressure amplitude (the parameter we differentiate)
σ_blast = 0.09
blast_xc, blast_yc = 0.25, 0.50

ε_fd     = 1.0e-3          # central finite-difference step
N_e_list = [16, 22, 30, 40, 54, 72]

@printf("OtD-gap refinement study (Oseen pair + blast)\n")
@printf("  fixed: p=%d, T=%.2f, α=%.4e, n_iter=%d, solver=%s, objective=%s\n",
        p, T, α, n_iter, solver, objective)
@printf("  window: xc=%.2f, yc=%.2f, σ=%.2f\n", KE_WINDOW.xc, KE_WINDOW.yc, KE_WINDOW.σ)
@printf("  directional derivative: blast amplitude (A=%.2f)\n\n", blast_A)

# ============================================================
# Helpers (parameterized by mesh so they work at any resolution)
# ============================================================

function build_ic(mesh, A)
    μx0 = zeros(size(mesh.x)); μy0 = zeros(size(mesh.x))
    ρ0  = similar(mesh.x);     E0  = similar(mesh.x)
    build_oseen_pair_blast_ic!(μx0, μy0, ρ0, E0, A, mesh, γ;
                               blast_xc=blast_xc, blast_yc=blast_yc, σ=σ_blast)
    return μx0, μy0, ρ0, E0
end

# CFL time step from the IC at amplitude A (kept FIXED across the FD stencil)
function timestep(mesh, μx0, μy0, ρ0, E0)
    mw = 0.0
    for idx in eachindex(μx0)
        mw = max(mw, max_wavespeed_x(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]),
                     max_wavespeed_y(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]))
    end
    Δt = CFL * min(mesh.Δx, mesh.Δy) / ((2*p + 1) * mw)
    n_steps = ceil(Int, T / Δt)
    return T / n_steps
end

# Objective J = windowed kinetic energy at T, for amplitude `A`,
# at a fixed time step `Δt`.
function objective_J(basis, mesh, Δt, A)
    n_p = basis.p + 1
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    μx0, μy0, ρ0, E0 = build_ic(mesh, A)
    μxf, μyf, ρf, _, _ = run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α;
                                     n_iter=n_iter, solver=solver)
    J = 0.0
    for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
        w = ke_window(mesh.x[i,j,ex,ey], mesh.y[i,j,ex,ey])
        wJ = basis.w[i]*basis.w[j]*mesh.Jx*mesh.Jy
        J += wJ * w * (μxf[i,j,ex,ey]^2 + μyf[i,j,ex,ey]^2) / (2*ρf[i,j,ex,ey])
    end
    return J
end

# ============================================================
# Refinement sweep
# ============================================================
hs   = Float64[]
adjs = Float64[]
fds  = Float64[]
gaps = Float64[]

println("  N_e     h          adjoint dJ/dA       FD dJ/dA        OtD gap")
for N_e in N_e_list
    basis = DGBasis(p)
    mesh  = CartesianMesh2D(N_e, N_e, L, L, basis)
    n_p   = p + 1

    μx0, μy0, ρ0, E0 = build_ic(mesh, blast_A)
    Δt = timestep(mesh, μx0, μy0, ρ0, E0)

    # --- adjoint directional derivative ---
    snapshots, n_steps_fwd = run_forward_store(μx0, μy0, ρ0, E0, Δt, T,
        basis, mesh, γ, α; n_iter=n_iter, solver=solver)
    _, _, _, aE = run_adjoint_conservative(snapshots, n_steps_fwd, Δt, T,
        basis, mesh, γ, α; n_iter=n_iter, objective=objective, solver=solver)

    dJ_adj = 0.0
    for ey in 1:N_e, ex in 1:N_e, j in 1:n_p, i in 1:n_p
        r2 = (mesh.x[i,j,ex,ey] - blast_xc)^2 + (mesh.y[i,j,ex,ey] - blast_yc)^2
        dE0_dA = exp(-r2 / σ_blast^2) / (γ - 1)
        wJ = basis.w[i]*basis.w[j]*mesh.Jx*mesh.Jy
        dJ_adj += wJ * aE[i,j,ex,ey] * dE0_dA
    end

    # --- central finite-difference directional derivative ---
    dJ_fd = (objective_J(basis, mesh, Δt, blast_A + ε_fd) -
             objective_J(basis, mesh, Δt, blast_A - ε_fd)) / (2 * ε_fd)

    gap = abs(dJ_adj - dJ_fd) / max(abs(dJ_fd), 1e-30)
    push!(hs, L / N_e); push!(adjs, dJ_adj); push!(fds, dJ_fd); push!(gaps, gap)
    @printf("  %3d    %.5f    %14.6e   %14.6e    %.3e\n", N_e, L/N_e, dJ_adj, dJ_fd, gap)
    flush(stdout)
end

# Observed convergence rates of the gap (gap ~ hʳ)
println("\n  observed OtD-gap convergence rates (gap ~ hʳ):")
for i in 2:length(N_e_list)
    r = log(gaps[i-1] / gaps[i]) / log(hs[i-1] / hs[i])
    @printf("    N_e %d → %d :  r = %.2f\n", N_e_list[i-1], N_e_list[i], r)
end

# ============================================================
# Export CSV
# ============================================================
out = joinpath(DATA_DIR, "otd_refinement_oseen.csv")
open(out, "w") do io
    println(io, "h,gap,adjoint,fd")
    for i in eachindex(hs)
        @printf(io, "%.10e,%.10e,%.10e,%.10e\n", hs[i], gaps[i], adjs[i], fds[i])
    end
end

println("\nData written to: $out")
