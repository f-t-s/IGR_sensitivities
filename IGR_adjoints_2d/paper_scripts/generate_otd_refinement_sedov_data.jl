#!/usr/bin/env julia
# ============================================================
# Generate CSV data for the optimize-then-discretize (OtD) gap
# mesh-refinement study on the three-Sedov-blast problem.
#
# At each mesh resolution N_e (with Δt ∝ h via a fixed CFL) we
# compute one directional derivative dJ/dA two ways:
#   - adjoint:  ⟨a†, ∂q₀/∂A⟩   (continuous PDE adjoint)
#   - FD:       central finite difference of the discrete forward map
# Their relative difference is the OtD gap. As the discretization is
# refined the gap should shrink, since both converge to the true
# derivative of the (fixed) continuous problem.
#
# All physical settings — including α — are held FIXED across the
# sweep, so refinement targets a single continuous problem.
#
# Output (in figures/data/):
#   otd_refinement_sedov.csv  — h, gap, adjoint, fd
# ============================================================

using Printf, LinearAlgebra

include(joinpath(@__DIR__, "..", "IGRAdjoints2D.jl"))
using .IGRAdjoints2D

# ============================================================
# Output directory
# ============================================================
# Figure data is written here, inside this repository. Copy the contents of
# paper_data/ into the paper's figures/data/ directory to rebuild the figures.
const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper_data")
mkpath(DATA_DIR)

# ============================================================
# Fixed settings (identical for every resolution)
# ============================================================
γ   = 1.4
L   = 1.0
p   = 4
T   = 0.10
CFL = 0.4
α   = 3.0 * (L / 20)^2     # FIXED physical IGR regularization (no mesh dependence)
n_iter = 60                # PCG iterations — enough to converge the elliptic solves
solver = :pcg
objective = :kinetic_energy

# Three blasts: centers and (slightly different) over-pressure amplitudes
blast_xc = [0.30, 0.70, 0.52]
blast_yc = [0.34, 0.40, 0.74]
blast_A  = [6.0,  5.0,  5.5]
σ_blast  = 0.09
p_bg     = 1.0
n_blast  = length(blast_A)

test_blast = 1             # which blast amplitude to differentiate
ε_fd       = 5.0e-4        # central finite-difference step
N_e_list   = [12, 16, 22, 30]

@printf("OtD-gap refinement study (three Sedov blasts)\n")
@printf("  fixed: p=%d, T=%.2f, α=%.4e, n_iter=%d, solver=%s, objective=%s\n",
        p, T, α, n_iter, solver, objective)
@printf("  directional derivative: amplitude of blast %d (A=%.2f)\n\n",
        test_blast, blast_A[test_blast])

# ============================================================
# Helpers (parameterized by mesh so they work at any resolution)
# ============================================================

# Three-blast IC on a given mesh, with amplitudes `Avec`.
function build_ic(mesh, Avec)
    μx0 = zeros(size(mesh.x)); μy0 = zeros(size(mesh.x))
    ρ0  = similar(mesh.x);     E0  = similar(mesh.x)
    for idx in eachindex(mesh.x)
        x = mesh.x[idx]; y = mesh.y[idx]
        pv = p_bg
        for k in 1:n_blast
            r2 = (x - blast_xc[k])^2 + (y - blast_yc[k])^2
            pv += Avec[k] * exp(-r2 / σ_blast^2)
        end
        ρ0[idx] = 1.0
        E0[idx] = pv / (γ - 1)
    end
    return μx0, μy0, ρ0, E0
end

# CFL-limited time step for an IC on a given mesh.
function timestep(mesh, μx0, μy0, ρ0, E0)
    mw = 0.0
    for idx in eachindex(μx0)
        mw = max(mw, max(max_wavespeed_x(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]),
                         max_wavespeed_y(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx])))
    end
    Δt = CFL * min(mesh.Δx, mesh.Δy) / ((2*p + 1) * mw)
    n_steps = ceil(Int, T / Δt)
    return T / n_steps
end

# Objective J = mass-averaged kinetic energy at T, for amplitudes `Avec`,
# at a fixed time step `Δt`.
function objective_J(basis, mesh, Δt, Avec)
    n_p = basis.p + 1
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    μx0, μy0, ρ0, E0 = build_ic(mesh, Avec)
    μxf, μyf, ρf, _, _ = run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α;
                                     n_iter=n_iter, solver=solver)
    KE = 0.0; Mtot = 0.0
    for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
        wJ = basis.w[i]*basis.w[j]*mesh.Jx*mesh.Jy
        KE   += wJ * (μxf[i,j,ex,ey]^2 + μyf[i,j,ex,ey]^2) / (2*ρf[i,j,ex,ey])
        Mtot += wJ * ρf[i,j,ex,ey]
    end
    return KE / Mtot
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
        r2 = (mesh.x[i,j,ex,ey] - blast_xc[test_blast])^2 +
             (mesh.y[i,j,ex,ey] - blast_yc[test_blast])^2
        dE0_dA = exp(-r2 / σ_blast^2) / (γ - 1)
        wJ = basis.w[i]*basis.w[j]*mesh.Jx*mesh.Jy
        dJ_adj += wJ * aE[i,j,ex,ey] * dE0_dA
    end

    # --- central finite-difference directional derivative ---
    Ap = copy(blast_A); Ap[test_blast] += ε_fd
    Am = copy(blast_A); Am[test_blast] -= ε_fd
    dJ_fd = (objective_J(basis, mesh, Δt, Ap) -
             objective_J(basis, mesh, Δt, Am)) / (2 * ε_fd)

    gap = abs(dJ_adj - dJ_fd) / max(abs(dJ_fd), 1e-30)
    push!(hs, L / N_e); push!(adjs, dJ_adj); push!(fds, dJ_fd); push!(gaps, gap)
    @printf("  %3d    %.5f    %14.6e   %14.6e    %.3e\n", N_e, L/N_e, dJ_adj, dJ_fd, gap)
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
out = joinpath(DATA_DIR, "otd_refinement_sedov.csv")
open(out, "w") do io
    println(io, "h,gap,adjoint,fd")
    for i in eachindex(hs)
        @printf(io, "%.10e,%.10e,%.10e,%.10e\n", hs[i], gaps[i], adjs[i], fds[i])
    end
end

println("\nData written to: $out")
