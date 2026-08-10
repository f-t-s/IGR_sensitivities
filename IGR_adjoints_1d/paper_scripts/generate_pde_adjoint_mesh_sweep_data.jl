#!/usr/bin/env julia
# ============================================================
# Generate CSV data for PDE adjoint mesh-resolution sweep
# (no SIAC filtering, no Enzyme comparison).
#
# Sweeps N_e ∈ {128, ..., 2048} for two anchored pairs of scaling
# designs, s ∈ {2, 4}:
#   linear:    √α = s·Δx            (α = s²·(L/N_e)²,   cells/layer fixed)
#   sublinear: √α = s·Δx₀^{1/3}Δx^{2/3}  (α = s²·128^{-2/3}·(L/N_e)^{4/3})
# Each pair shares its α at the coarsest level N_e = 128 (Δx₀ = L/128),
# then refines along different paths: the linear path keeps the number
# of cells per layer width √α fixed, while the sublinear path resolves
# the layers asymptotically (cells/layer ∝ N_e^{1/3}, cf. Giles &
# Ulbrich). Runs the forward solve and the PDE adjoint, and writes
# per-resolution CSVs of:
#   - the primal at t = T  in primitive variables (u, ρ, p, Σ)
#   - the PDE adjoint at t = 0 in primitive variables
#     (∂J/∂u₀, ∂J/∂ρ₀, ∂J/∂p₀)
#
# Outputs (in figures/data/), per N_e:
#   pde_mesh_sweep_{lin2,gu2p,lin4,gu4p}_Ne{N}_primal.csv
#   pde_mesh_sweep_{lin2,gu2p,lin4,gu4p}_Ne{N}_adj_pde.csv
# ============================================================

using Printf, LinearAlgebra

include(joinpath(@__DIR__, "..", "IGRAdjoints1D.jl"))
using .IGRAdjoints1D

# ============================================================
# Output directory
# ============================================================
# Figure data is written here, inside this repository. Copy the contents of
# paper_data/ into the paper's figures/data/ directory to rebuild the figures.
const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper_data")
mkpath(DATA_DIR)

# ============================================================
# Helpers (same as the other paper scripts)
# ============================================================

function lagrange_interpolation_matrix(ξ_from, ξ_to)
    n = length(ξ_from); m = length(ξ_to)
    λ = ones(n)
    for j in 1:n
        for i in 1:n
            i == j && continue
            λ[j] *= (ξ_from[j] - ξ_from[i])
        end
        λ[j] = 1.0 / λ[j]
    end
    V = zeros(m, n)
    for k in 1:m
        x = ξ_to[k]
        exact = 0
        for j in 1:n
            if abs(x - ξ_from[j]) < 1e-14; exact = j; break; end
        end
        if exact > 0
            V[k, exact] = 1.0
        else
            denom = 0.0
            for j in 1:n
                t = λ[j] / (x - ξ_from[j]); V[k, j] = t; denom += t
            end
            V[k, :] ./= denom
        end
    end
    return V
end

function flatten_dg(mesh, basis, q; n_sub=10)
    n_p, N_e = size(q)
    ξ_fine = collect(range(-1, 1, length=n_sub))
    V = lagrange_interpolation_matrix(basis.ξ, ξ_fine)
    x_out = Float64[]
    q_out = Float64[]
    for e in 1:N_e
        x_lo = mesh.x[1, e]; x_hi = mesh.x[n_p, e]
        x_mid = (x_lo + x_hi) / 2; x_half = (x_hi - x_lo) / 2
        q_fine = V * q[:, e]
        for k in 1:n_sub
            push!(x_out, x_mid + x_half * ξ_fine[k])
            push!(q_out, q_fine[k])
        end
        if e < N_e
            push!(x_out, NaN); push!(q_out, NaN)
        end
    end
    return x_out, q_out
end

# ============================================================
# Generate data for one mesh resolution
# ============================================================

function generate_mesh_sweep_case(;
        prefix,
        N_e,
        objective = :weighted_momentum,
        A         = 1.5,
        p         = 4,
        γ         = 1.4,
        L         = 1.0,
        T         = 0.25,
        CFL       = 1.0,
        α_scale   = 9.0,
        α_exp     = 2.0,
        n_iter    = 102,
        solver    = :pcg,
    )

    α = α_scale * (L / N_e)^α_exp
    basis = DGBasis(p)
    mesh  = PeriodicMesh1D(N_e, L, basis)
    n_p   = p + 1

    max_ws_est = 2.5 + abs(A)
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)
    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    @printf("[%s] obj=%s, p=%d, N_e=%d, α=%.4e, T=%.3f, n_steps=%d, n_iter=%d, solver=%s\n",
            prefix, objective, p, N_e, α, T, n_steps, n_iter, solver)

    # Build IC (velocity pulse)
    μ0 = zeros(n_p, N_e); ρ0 = zeros(n_p, N_e); E0 = zeros(n_p, N_e)
    build_velocity_pulse_ic!(μ0, ρ0, E0, A, mesh.x, L, γ)

    # ----------------------------------------------------------
    # Forward solve (store snapshots)
    # ----------------------------------------------------------
    println("  forward solve...")
    snapshots, n_steps_fwd = run_forward_store(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α;
                                                n_iter=n_iter, solver=solver)

    # ----------------------------------------------------------
    # PDE adjoint (optimize-then-discretize) → conservative L² adjoints
    # ----------------------------------------------------------
    println("  PDE adjoint...")
    aμ, aρ, aE = run_adjoint_conservative(snapshots, n_steps_fwd, Δt, T,
        basis, mesh, γ, α; n_iter=n_iter, objective=objective, solver=solver)

    # ----------------------------------------------------------
    # Primal at T (primitive variables)
    # ----------------------------------------------------------
    μT, ρT, ET, ΣT = snapshots[end]
    uT = μT ./ ρT
    pT = (γ - 1) .* (ET .- μT.^2 ./ (2 .* ρT))

    # ----------------------------------------------------------
    # Conservative adjoints → primitive (u, ρ, p) at t = 0
    #   ∂J/∂u₀ = ρ₀·aμ + ρ₀·u₀·aE
    #   ∂J/∂ρ₀ = aρ + u₀·aμ + (u₀²/2)·aE
    #   ∂J/∂p₀ = aE / (γ - 1)
    # ----------------------------------------------------------
    u0    = μ0 ./ ρ0
    adj_u = ρ0 .* aμ .+ ρ0 .* u0 .* aE
    adj_ρ = aρ .+ u0 .* aμ .+ (u0.^2 ./ 2) .* aE
    adj_p = aE ./ (γ - 1)

    # ----------------------------------------------------------
    # Export CSVs
    # ----------------------------------------------------------
    println("  writing CSVs...")
    xs, _ = flatten_dg(mesh, basis, uT)

    _, u_s = flatten_dg(mesh, basis, uT)
    _, ρ_s = flatten_dg(mesh, basis, ρT)
    _, p_s = flatten_dg(mesh, basis, pT)
    _, Σ_s = flatten_dg(mesh, basis, ΣT)
    open(joinpath(DATA_DIR, "$(prefix)_primal.csv"), "w") do io
        println(io, "x,u,rho,p,sigma")
        for i in eachindex(xs)
            @printf(io, "%.10e,%.10e,%.10e,%.10e,%.10e\n",
                    xs[i], u_s[i], ρ_s[i], p_s[i], Σ_s[i])
        end
    end

    _, au_s = flatten_dg(mesh, basis, adj_u)
    _, aρ_s = flatten_dg(mesh, basis, adj_ρ)
    _, ap_s = flatten_dg(mesh, basis, adj_p)
    open(joinpath(DATA_DIR, "$(prefix)_adj_pde.csv"), "w") do io
        println(io, "x,adj_u,adj_rho,adj_p")
        for i in eachindex(xs)
            @printf(io, "%.10e,%.10e,%.10e,%.10e\n",
                    xs[i], au_s[i], aρ_s[i], ap_s[i])
        end
    end

    # Free memory before the next resolution
    for s in snapshots
        fill!(s[1], 0); fill!(s[2], 0); fill!(s[3], 0); fill!(s[4], 0)
    end
    empty!(snapshots)
    fill!(aμ, 0); fill!(aρ, 0); fill!(aE, 0)
    fill!(μ0, 0); fill!(ρ0, 0); fill!(E0, 0)
    GC.gc()

    return nothing
end

# ============================================================
# Run cases: sweep over N_e
# ============================================================
const N_E_LIST = [128, 256, 512, 1024, 2048, 4096]

for N_e in N_E_LIST
    tag = @sprintf("Ne%d", N_e)
    println("="^60)
    println("PDE adjoint mesh sweep — N_e = $N_e")
    println("="^60)
    # Anchored pairs: the sublinear (GU) partner shares the linear design's
    # α at N_e = 128:  s_gu² = s_lin² · 128^{-2/3}.
    for (s2, ex, sfx) in ((4.0, 2.0, "lin2"), (16.0, 2.0, "lin4"),
                          (4.0 / 128.0^(2/3), 4/3, "gu2p"),
                          (16.0 / 128.0^(2/3), 4/3, "gu4p"))
        generate_mesh_sweep_case(prefix="pde_mesh_sweep_$(sfx)_$tag", N_e=N_e, α_scale=s2, α_exp=ex)
    end
    println()
end

println("All data written to: $DATA_DIR")
