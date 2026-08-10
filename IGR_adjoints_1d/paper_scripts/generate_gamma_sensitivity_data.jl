#!/usr/bin/env julia
# ============================================================
# Generate CSV data for the γ-sensitivity experiment
#
# Two interacting Sedov-like blasts on a periodic domain.
# Computes ∂q/∂γ at constant initial pressure profile.
#
# Methods compared:
#   1. ForwardDiff (AD through discrete operator)
#   2. Finite differences (central)
#   3. PDE-based sensitivity equations (requires ∂F/∂γ source)
#
# Outputs CSV files to the paper figures/data/ directory.
# ============================================================

using Printf, LinearAlgebra

include(joinpath(@__DIR__, "..", "IGRAdjoints1D.jl"))
using .IGRAdjoints1D

using ForwardDiff

# ============================================================
# Output directory
# ============================================================
# Figure data is written here, inside this repository. Copy the contents of
# paper_data/ into the paper's figures/data/ directory to rebuild the figures.
const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper_data")
mkpath(DATA_DIR)

# ============================================================
# Helpers
# ============================================================

"""
    double_blast_ic(mesh, γ; p_bg, p1, p2, x1, x2, σ)

Two Sedov-like blasts on a periodic domain:
  ρ = 1, u = 0, p = p_bg + p1·exp(-(x-x1)²/σ²) + p2·exp(-(x-x2)²/σ²)

γ can be a ForwardDiff Dual for AD.
All returned arrays have the same eltype as γ.
"""
function double_blast_ic(mesh, γ;
        p_bg = 1.0, p1 = 100.0, p2 = 50.0,
        x1 = 0.3, x2 = 0.7, σ = 0.02)
    n_p, N_e = size(mesh.x)
    L = mesh.L
    RT = typeof(γ)

    μ0 = zeros(RT, n_p, N_e)
    ρ0 = zeros(RT, n_p, N_e)
    E0 = zeros(RT, n_p, N_e)

    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i, e]
        ρ_val = one(RT)
        u_val = zero(RT)
        p_val = RT(p_bg) + RT(p1) * exp(-((x - x1) / σ)^2) +
                            RT(p2) * exp(-((x - x2) / σ)^2)
        ρ0[i, e] = ρ_val
        μ0[i, e] = ρ_val * u_val
        E0[i, e] = p_val / (γ - 1) + ρ_val * u_val^2 / 2
    end
    return μ0, ρ0, E0
end

"""
    gamma_sensitivity_ic(mesh, γ; kwargs...)

Compute ∂q₀/∂γ at constant pressure profile.
Since ρ₀=1, u₀=0, E₀ = p₀/(γ-1):
  ∂ρ₀/∂γ = 0, ∂μ₀/∂γ = 0, ∂E₀/∂γ = -p₀/(γ-1)²
"""
function gamma_sensitivity_ic(mesh, γ;
        p_bg = 1.0, p1 = 100.0, p2 = 50.0,
        x1 = 0.3, x2 = 0.7, σ = 0.02)
    n_p, N_e = size(mesh.x)

    sμ0 = zeros(n_p, N_e)
    sρ0 = zeros(n_p, N_e)
    sE0 = zeros(n_p, N_e)

    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i, e]
        p_val = p_bg + p1 * exp(-((x - x1) / σ)^2) +
                        p2 * exp(-((x - x2) / σ)^2)
        sE0[i, e] = -p_val / (γ - 1)^2
    end
    return sμ0, sρ0, sE0
end

"""
    lagrange_interpolation_matrix(ξ_from, ξ_to)

Interpolation matrix using barycentric Lagrange interpolation.
"""
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

"""
    flatten_dg(mesh, basis, q; n_sub=10)

Flatten (n_p × N_e) DG field to 1D vectors for CSV output,
evaluating the polynomial on `n_sub` equispaced points per element.
NaN separators between elements cause pgfplots to break the line,
showing individual DG hat functions without vertical jumps.
Returns (x_out, q_out).
"""
function flatten_dg(mesh, basis, q; n_sub=10)
    n_p, N_e = size(q)
    ξ_fine = collect(range(-1, 1, length=n_sub))
    V = lagrange_interpolation_matrix(basis.ξ, ξ_fine)

    x_out = Float64[]
    q_out = Float64[]
    for e in 1:N_e
        x_lo = mesh.x[1, e]
        x_hi = mesh.x[n_p, e]
        x_mid = (x_lo + x_hi) / 2
        x_half = (x_hi - x_lo) / 2
        q_fine = V * q[:, e]
        for k in 1:n_sub
            push!(x_out, x_mid + x_half * ξ_fine[k])
            push!(q_out, q_fine[k])
        end
        if e < N_e
            push!(x_out, NaN)
            push!(q_out, NaN)
        end
    end
    return x_out, q_out
end

"""
    conservative_to_primitive(μ, ρ, E, γ)

Convert conservative (μ, ρ, E) to primitive (u, ρ, p) fields.
"""
function conservative_to_primitive(μ, ρ, E, γ)
    u = μ ./ ρ
    p = (γ - 1) .* (E .- μ.^2 ./ (2 .* ρ))
    return u, ρ, p
end

# ============================================================
# Generate data for one time
# ============================================================

function generate_case(;
        prefix,
        γ = 1.4,
        p_order = 2,
        N_e = 256,
        L = 1.0,
        T,
        CFL = 0.3,
        α_scale = 1.0,
        n_iter = 5,
        fd_eps = 1e-6,
        solver = :chebyshev,
        # Blast parameters
        p_bg = 1.0, p1 = 100.0, p2 = 50.0,
        x1 = 0.3, x2 = 0.7, σ = 0.02,
    )

    α = α_scale * (L / N_e)^2
    basis = DGBasis(p_order)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p_order + 1

    blast_kw = (p_bg=p_bg, p1=p1, p2=p2, x1=x1, x2=x2, σ=σ)

    # Estimate max wave speed from blast pressure
    c_max = sqrt(γ * (p_bg + max(p1, p2)))  # sound speed at peak
    max_ws_est = c_max + 1.0  # safety margin
    Δt = CFL * mesh.Δx / ((2*p_order + 1) * max_ws_est)

    @printf("  [%s] p=%d, N_e=%d, α=%.4e, T=%.4f, γ=%.2f\n",
            prefix, p_order, N_e, α, T, γ)

    # ----------------------------------------------------------
    # 1. ForwardDiff: differentiate w.r.t. γ
    # ----------------------------------------------------------
    println("    ForwardDiff...")
    γ_dual = ForwardDiff.Dual(γ, 1.0)
    μ0_d, ρ0_d, E0_d = double_blast_ic(mesh, γ_dual; blast_kw...)
    μf_d, ρf_d, Ef_d, Σf_d = run_forward(μ0_d, ρ0_d, E0_d, Δt, T, basis, mesh, γ_dual, α;
                                        n_iter=n_iter, solver=solver)

    μf_val = ForwardDiff.value.(μf_d)
    ρf_val = ForwardDiff.value.(ρf_d)
    Ef_val = ForwardDiff.value.(Ef_d)
    Σf_val = ForwardDiff.value.(Σf_d)

    uf_val, _, pf_val = conservative_to_primitive(μf_val, ρf_val, Ef_val, γ)

    sμ_ad = ForwardDiff.partials.(μf_d, 1)
    sρ_ad = ForwardDiff.partials.(ρf_d, 1)
    sE_ad = ForwardDiff.partials.(Ef_d, 1)
    sΣ_ad = ForwardDiff.partials.(Σf_d, 1)

    u_ref = μf_val ./ ρf_val
    su_ad = (sμ_ad .- u_ref .* sρ_ad) ./ ρf_val
    sp_ad = (γ - 1) .* (sE_ad .- sρ_ad .* u_ref.^2 ./ 2 .- ρf_val .* u_ref .* su_ad) .+
            (Ef_val .- μf_val.^2 ./ (2 .* ρf_val))  # + ∂p/∂γ|_q = E - ½ρu²

    # ----------------------------------------------------------
    # 2. Finite differences (central)
    # ----------------------------------------------------------
    println("    Finite differences...")
    μ0_p, ρ0_p, E0_p = double_blast_ic(mesh, γ + fd_eps; blast_kw...)
    μ0_m, ρ0_m, E0_m = double_blast_ic(mesh, γ - fd_eps; blast_kw...)

    μf_p, ρf_p, Ef_p, Σf_p = run_forward(μ0_p, ρ0_p, E0_p, Δt, T, basis, mesh, γ + fd_eps, α;
                                        n_iter=n_iter, solver=solver)
    μf_m, ρf_m, Ef_m, Σf_m = run_forward(μ0_m, ρ0_m, E0_m, Δt, T, basis, mesh, γ - fd_eps, α;
                                        n_iter=n_iter, solver=solver)

    sμ_fd = (μf_p .- μf_m) ./ (2 * fd_eps)
    sρ_fd = (ρf_p .- ρf_m) ./ (2 * fd_eps)
    sE_fd = (Ef_p .- Ef_m) ./ (2 * fd_eps)
    sΣ_fd = (Σf_p .- Σf_m) ./ (2 * fd_eps)

    uf_p, _, pf_p = conservative_to_primitive(μf_p, ρf_p, Ef_p, γ + fd_eps)
    uf_m, _, pf_m = conservative_to_primitive(μf_m, ρf_m, Ef_m, γ - fd_eps)
    su_fd = (uf_p .- uf_m) ./ (2 * fd_eps)
    sp_fd = (pf_p .- pf_m) ./ (2 * fd_eps)

    # ----------------------------------------------------------
    # 3. PDE-based sensitivity equations
    #
    # sγ=1.0 activates the ∂P/∂γ = ρe source term in the
    # sensitivity flux (adjoints.tex eq. 149).
    # ----------------------------------------------------------
    println("    PDE sensitivity...")
    μ0, ρ0, E0 = double_blast_ic(mesh, γ; blast_kw...)
    sμ0, sρ0, sE0 = gamma_sensitivity_ic(mesh, γ; blast_kw...)

    μf_pde, ρf_pde, Ef_pde, Σf_pde, sμ_pde, sρ_pde, sE_pde, sΣ_pde =
        run_sensitivity(μ0, ρ0, E0, sμ0, sρ0, sE0, Δt, T, basis, mesh, γ, α;
                        n_iter=n_iter, solver=solver, sγ=1.0)

    su_pde = (sμ_pde .- u_ref .* sρ_pde) ./ ρf_val
    sp_pde = (γ - 1) .* (sE_pde .- sρ_pde .* u_ref.^2 ./ 2 .- ρf_val .* u_ref .* su_pde) .+
             (Ef_val .- μf_val.^2 ./ (2 .* ρf_val))

    # ----------------------------------------------------------
    # Export CSV files
    # ----------------------------------------------------------
    println("    Writing CSVs...")

    xs, _ = flatten_dg(mesh, basis, uf_val)

    _, u_s = flatten_dg(mesh, basis, uf_val)
    _, ρ_s = flatten_dg(mesh, basis, ρf_val)
    _, p_s = flatten_dg(mesh, basis, pf_val)
    _, Σ_s = flatten_dg(mesh, basis, Σf_val)

    open(joinpath(DATA_DIR, "$(prefix)_primal.csv"), "w") do io
        println(io, "x,u,rho,p,sigma")
        for i in eachindex(xs)
            @printf(io, "%.10e,%.10e,%.10e,%.10e,%.10e\n", xs[i], u_s[i], ρ_s[i], p_s[i], Σ_s[i])
        end
    end

    function write_sensitivity_csv(filename, su, sρ, sp, sΣ)
        _, su_s = flatten_dg(mesh, basis, su)
        _, sρ_s = flatten_dg(mesh, basis, sρ)
        _, sp_s = flatten_dg(mesh, basis, sp)
        _, sΣ_s = flatten_dg(mesh, basis, sΣ)
        open(joinpath(DATA_DIR, filename), "w") do io
            println(io, "x,su,srho,sp,ssigma")
            for i in eachindex(xs)
                @printf(io, "%.10e,%.10e,%.10e,%.10e,%.10e\n", xs[i], su_s[i], sρ_s[i], sp_s[i], sΣ_s[i])
            end
        end
    end

    write_sensitivity_csv("$(prefix)_sens_ad.csv", su_ad, sρ_ad, sp_ad, sΣ_ad)
    write_sensitivity_csv("$(prefix)_sens_fd.csv", su_fd, sρ_fd, sp_fd, sΣ_fd)
    write_sensitivity_csv("$(prefix)_sens_pde.csv", su_pde, sρ_pde, sp_pde, sΣ_pde)

    # ----------------------------------------------------------
    # Print summary errors
    # ----------------------------------------------------------
    println("    Relative L2 errors vs ForwardDiff:")
    for (label, su_r, sρ_r, sp_r) in [
            ("FD         ", su_fd, sρ_fd, sp_fd),
            ("PDE        ", su_pde, sρ_pde, sp_pde)]
        err_u = norm(su_r .- su_ad) / max(norm(su_ad), 1e-15)
        err_ρ = norm(sρ_r .- sρ_ad) / max(norm(sρ_ad), 1e-15)
        err_p = norm(sp_r .- sp_ad) / max(norm(sp_ad), 1e-15)
        @printf("      %s:  u: %.4e   ρ: %.4e   p: %.4e\n", label, err_u, err_ρ, err_p)
    end

    return nothing
end

# ============================================================
# Run cases
# ============================================================

params = (γ=1.4, p_order=2, N_e=64, L=1.0, CFL=0.5,
          α_scale=10.0, n_iter=50, fd_eps=1e-6, solver=:jacobi,
          p_bg=1.0, p1=100.0, p2=50.0, x1=0.3, x2=0.7, σ=0.1)

# Late time: after blasts have interacted
println("="^60)
println("Late time (T = 0.05)")
println("="^60)
generate_case(; prefix="blast_late", T=0.05, params...)

println("\nAll data written to: $DATA_DIR")
