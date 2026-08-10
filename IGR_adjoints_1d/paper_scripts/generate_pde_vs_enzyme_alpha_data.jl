#!/usr/bin/env julia
# ============================================================
# Generate CSV data for PDE adjoint vs Enzyme reverse-mode AD
# at three IGR regularization levels.
#
# Sweeps α = α_scale · (L/N_e)² for α_scale ∈ {1, 3, 9},
# otherwise identical to generate_pde_vs_enzyme_data.jl.
#
# Outputs (in figures/data/), per α_scale value:
#   pde_vs_enzyme_alpha{S}_primal.csv     — primal at T:  x, u, rho, p, eint, sigma
#   pde_vs_enzyme_alpha{S}_adj_pde.csv    — PDE adjoint:    x, adj_u, adj_rho, adj_eint
#   pde_vs_enzyme_alpha{S}_adj_enzyme.csv — Enzyme adjoint: x, adj_u, adj_rho, adj_eint
# ============================================================

using Printf, LinearAlgebra
using Enzyme

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
# Helpers (same as generate_adjoint_data.jl)
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
# Generate data for one α_scale
# ============================================================

function generate_pde_vs_enzyme_case(;
        prefix,
        α_scale,
        objective = :weighted_momentum,
        A         = 1.5,
        p         = 3,
        N_e       = 256,
        γ         = 1.4,
        L         = 1.0,
        T         = 0.25,
        CFL       = 0.5,
        n_iter    = 5,
        solver    = :chebyshev,
    )

    α = α_scale * (L / N_e)^2
    basis = DGBasis(p)
    mesh  = PeriodicMesh1D(N_e, L, basis)
    n_p   = p + 1

    max_ws_est = 2.5 + abs(A)
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)
    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    @printf("[%s] obj=%s, p=%d, N_e=%d, α_scale=%.1f, α=%.4e, T=%.3f, n_steps=%d, n_iter=%d, solver=%s\n",
            prefix, objective, p, N_e, α_scale, α, T, n_steps, n_iter, solver)

    # Build IC (velocity pulse)
    μ0 = zeros(n_p, N_e); ρ0 = zeros(n_p, N_e); E0 = zeros(n_p, N_e)
    build_velocity_pulse_ic!(μ0, ρ0, E0, A, mesh.x, L, γ)

    # ----------------------------------------------------------
    # Select objective function for Enzyme
    # ----------------------------------------------------------
    obj_fn = if objective == :l2_density
        IGRAdjoints1D._enzyme_objective
    elseif objective == :l2_pressure
        IGRAdjoints1D._enzyme_objective_l2_pressure
    elseif objective == :kinetic_energy
        IGRAdjoints1D._enzyme_objective_kinetic_energy
    elseif objective == :weighted_momentum
        IGRAdjoints1D._enzyme_objective_weighted_momentum
    else
        error("Unknown objective: $objective")
    end

    # ----------------------------------------------------------
    # 1. Enzyme reverse-mode AD: ∂J/∂q₀
    # ----------------------------------------------------------
    println("  Enzyme AD...")
    dμ0_enz = zeros(n_p, N_e); dρ0_enz = zeros(n_p, N_e); dE0_enz = zeros(n_p, N_e)

    mode = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
    result = Enzyme.autodiff(mode, obj_fn, Enzyme.Active,
        Enzyme.Const(α),
        Enzyme.Duplicated(copy(μ0), dμ0_enz),
        Enzyme.Duplicated(copy(ρ0), dρ0_enz),
        Enzyme.Duplicated(copy(E0), dE0_enz),
        Enzyme.Const(n_p), Enzyme.Const(N_e),
        Enzyme.Const(n_steps), Enzyme.Const(Δt),
        Enzyme.Const(basis), Enzyme.Const(mesh),
        Enzyme.Const(γ), Enzyme.Const(max(n_iter, 1)), Enzyme.Const(solver))

    J_enz = result[2]
    @printf("  J (Enzyme) = %.10e\n", J_enz)

    # Convert Enzyme's discrete dJ/dq to L² density (divide by w_i · J)
    enz_cμ = similar(dμ0_enz); enz_cρ = similar(dρ0_enz); enz_cE = similar(dE0_enz)
    for e in 1:N_e, i in 1:n_p
        inv_wJ = 1.0 / (basis.w[i] * mesh.J)
        enz_cμ[i,e] = dμ0_enz[i,e] * inv_wJ
        enz_cρ[i,e] = dρ0_enz[i,e] * inv_wJ
        enz_cE[i,e] = dE0_enz[i,e] * inv_wJ
    end

    # ----------------------------------------------------------
    # 2. PDE adjoint (optimize-then-discretize)
    # ----------------------------------------------------------
    println("  PDE adjoint (forward + backward)...")
    snapshots, n_steps_fwd = run_forward_store(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α;
                                                n_iter=n_iter, solver=solver)
    pde_cμ, pde_cρ, pde_cE = run_adjoint_conservative(snapshots, n_steps_fwd, Δt, T,
        basis, mesh, γ, α; n_iter=n_iter, objective=objective, solver=solver)

    # ----------------------------------------------------------
    # Transform conservative L² adjoints → primitive (u, ρ, e_int)
    #   q†_u    = ρ q†_μ + ρu q†_E
    #   q†_ρ    = u q†_μ + q†_ρ + (e_int + u²/2) q†_E
    #   q†_eint = ρ q†_E
    # ----------------------------------------------------------
    u0    = μ0 ./ ρ0
    eint0 = E0 ./ ρ0 .- u0.^2 ./ 2

    enz_u    = ρ0 .* enz_cμ .+ ρ0 .* u0 .* enz_cE
    enz_ρ    = u0 .* enz_cμ .+ enz_cρ .+ (eint0 .+ u0.^2 ./ 2) .* enz_cE
    enz_eint = ρ0 .* enz_cE

    pde_u    = ρ0 .* pde_cμ .+ ρ0 .* u0 .* pde_cE
    pde_ρ    = u0 .* pde_cμ .+ pde_cρ .+ (eint0 .+ u0.^2 ./ 2) .* pde_cE
    pde_eint = ρ0 .* pde_cE

    # ----------------------------------------------------------
    # Primal at T (primitive variables)
    # ----------------------------------------------------------
    μT, ρT, ET, ΣT = snapshots[end]
    uT     = μT ./ ρT
    pT     = (γ - 1) .* (ET .- μT.^2 ./ (2 .* ρT))
    eintT  = ET ./ ρT .- uT.^2 ./ 2

    # ----------------------------------------------------------
    # Export CSVs
    # ----------------------------------------------------------
    println("  Writing CSVs...")
    xs, _ = flatten_dg(mesh, basis, uT)

    # Primal at final time
    _, u_s    = flatten_dg(mesh, basis, uT)
    _, ρ_s    = flatten_dg(mesh, basis, ρT)
    _, p_s    = flatten_dg(mesh, basis, pT)
    _, eint_s = flatten_dg(mesh, basis, eintT)
    _, Σ_s    = flatten_dg(mesh, basis, ΣT)
    open(joinpath(DATA_DIR, "$(prefix)_primal.csv"), "w") do io
        println(io, "x,u,rho,p,eint,sigma")
        for i in eachindex(xs)
            @printf(io, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                    xs[i], u_s[i], ρ_s[i], p_s[i], eint_s[i], Σ_s[i])
        end
    end

    # Adjoint fields in primitive variables
    function write_primitive_adjoint_csv(filename, adj_u, adj_ρ, adj_eint)
        _, au_s    = flatten_dg(mesh, basis, adj_u)
        _, aρ_s    = flatten_dg(mesh, basis, adj_ρ)
        _, aeint_s = flatten_dg(mesh, basis, adj_eint)
        open(joinpath(DATA_DIR, filename), "w") do io
            println(io, "x,adj_u,adj_rho,adj_eint")
            for i in eachindex(xs)
                @printf(io, "%.10e,%.10e,%.10e,%.10e\n",
                        xs[i], au_s[i], aρ_s[i], aeint_s[i])
            end
        end
    end

    write_primitive_adjoint_csv("$(prefix)_adj_enzyme.csv", enz_u, enz_ρ, enz_eint)
    write_primitive_adjoint_csv("$(prefix)_adj_pde.csv",    pde_u, pde_ρ, pde_eint)

    # ----------------------------------------------------------
    # Print comparison
    # ----------------------------------------------------------
    println("  Relative L² differences (PDE vs Enzyme), primitive variables:")
    err_u    = norm(pde_u    .- enz_u)    / max(norm(enz_u),    1e-15)
    err_ρ    = norm(pde_ρ    .- enz_ρ)    / max(norm(enz_ρ),    1e-15)
    err_eint = norm(pde_eint .- enz_eint) / max(norm(enz_eint), 1e-15)
    @printf("    u: %.4e   ρ: %.4e   e_int: %.4e\n", err_u, err_ρ, err_eint)

    return nothing
end

# ============================================================
# Run cases: α_scale ∈ {1, 3, 9}
# ============================================================
for α_scale in (1.0, 4.0, 9.0)
    tag = @sprintf("alpha%d", round(Int, α_scale))
    println("="^60)
    println("PDE adjoint vs Enzyme — α_scale = $α_scale")
    println("="^60)
    generate_pde_vs_enzyme_case(prefix="pde_vs_enzyme_$tag", α_scale=α_scale)
    println()
end

println("All data written to: $DATA_DIR")
