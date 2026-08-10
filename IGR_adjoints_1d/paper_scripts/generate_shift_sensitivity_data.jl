#!/usr/bin/env julia
# ============================================================
# Generate CSV data for the cyclic shift sensitivity experiment
#
# Compares three approaches for computing ∂q/∂s where s is a
# cyclic shift of the velocity sine wave IC:
#   1. ForwardDiff (AD through discrete operator)
#   2. Finite differences (central)
#   3. PDE-based sensitivity equations
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
    shifted_velocity_ic(mesh, γ, A, θ)

Velocity sine wave IC evaluated at x - θ:
  ρ = 1, u = A sin(2π(x-θ)/L), p = 1.
θ can be a ForwardDiff Dual for AD.
"""
function shifted_velocity_ic(mesh, γ, A, θ)
    n_p, N_e = size(mesh.x)
    L = mesh.L
    RT = promote_type(typeof(θ), typeof(A))
    μ0 = zeros(RT, n_p, N_e)
    ρ0 = zeros(RT, n_p, N_e)
    E0 = zeros(RT, n_p, N_e)

    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i,e] - θ
        ρ_val = one(RT)
        u_val = A * sin(2π * x / L)
        p_val = one(RT)
        ρ0[i,e] = ρ_val
        μ0[i,e] = ρ_val * u_val
        E0[i,e] = p_val / (γ - 1) + ρ_val * u_val^2 / 2
    end
    return μ0, ρ0, E0
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

"""
    init_shift_sensitivity_ic(mesh, basis, μ0, ρ0, E0)

Compute ∂q₀/∂s = -∂q₀/∂x using the DG derivative matrix.
"""
function init_shift_sensitivity_ic(mesh, basis, μ0, ρ0, E0)
    n_p, N_e = size(mesh.x)
    D = basis.D
    invJ = 1.0 / mesh.J

    sμ0 = zeros(n_p, N_e)
    sρ0 = zeros(n_p, N_e)
    sE0 = zeros(n_p, N_e)

    for e in 1:N_e, i in 1:n_p
        for j in 1:n_p
            sμ0[i,e] -= invJ * D[i,j] * μ0[j,e]
            sρ0[i,e] -= invJ * D[i,j] * ρ0[j,e]
            sE0[i,e] -= invJ * D[i,j] * E0[j,e]
        end
    end

    return sμ0, sρ0, sE0
end

# ============================================================
# Generate data for one case (pre-shock or post-shock)
# ============================================================

function generate_case(;
        prefix,
        A = 1.5,
        p = 2,
        N_e = 64,
        γ = 1.4,
        L = 1.0,
        T,
        CFL = 0.5,
        α_scale = 1.0,
        n_iter = 5,
        fd_eps = 1e-5,
        solver = :chebyshev,
    )

    α = α_scale * (L / N_e)^2
    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    max_ws_est = 2.5 + abs(A)
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)

    @printf("  [%s] p=%d, N_e=%d, α=%.4e, T=%.2f, A=%.2f\n", prefix, p, N_e, α, T, A)

    # ----------------------------------------------------------
    # 1. ForwardDiff
    # ----------------------------------------------------------
    println("    ForwardDiff...")
    θ_dual = ForwardDiff.Dual(0.0, 1.0)
    μ0_d, ρ0_d, E0_d = shifted_velocity_ic(mesh, γ, A, θ_dual)
    μf_d, ρf_d, Ef_d, Σf_d = run_forward(μ0_d, ρ0_d, E0_d, Δt, T, basis, mesh, γ, α;
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

    u_ad = μf_val ./ ρf_val
    su_ad = (sμ_ad .- u_ad .* sρ_ad) ./ ρf_val
    sp_ad = (γ - 1) .* (sE_ad .- sρ_ad .* u_ad.^2 ./ 2 .- ρf_val .* u_ad .* su_ad)

    # ----------------------------------------------------------
    # 2. Finite differences (central)
    # ----------------------------------------------------------
    println("    Finite differences...")
    μ0_p, ρ0_p, E0_p = shifted_velocity_ic(mesh, γ, A, +fd_eps)
    μ0_m, ρ0_m, E0_m = shifted_velocity_ic(mesh, γ, A, -fd_eps)

    μf_p, ρf_p, Ef_p, Σf_p = run_forward(μ0_p, ρ0_p, E0_p, Δt, T, basis, mesh, γ, α;
                                        n_iter=n_iter, solver=solver)
    μf_m, ρf_m, Ef_m, Σf_m = run_forward(μ0_m, ρ0_m, E0_m, Δt, T, basis, mesh, γ, α;
                                        n_iter=n_iter, solver=solver)

    sμ_fd = (μf_p .- μf_m) ./ (2 * fd_eps)
    sρ_fd = (ρf_p .- ρf_m) ./ (2 * fd_eps)
    sE_fd = (Ef_p .- Ef_m) ./ (2 * fd_eps)
    sΣ_fd = (Σf_p .- Σf_m) ./ (2 * fd_eps)

    su_fd = (sμ_fd .- u_ad .* sρ_fd) ./ ρf_val
    sp_fd = (γ - 1) .* (sE_fd .- sρ_fd .* u_ad.^2 ./ 2 .- ρf_val .* u_ad .* su_fd)

    # ----------------------------------------------------------
    # 3. PDE-based sensitivity equations
    # ----------------------------------------------------------
    println("    PDE sensitivity...")
    μ0, ρ0, E0 = shifted_velocity_ic(mesh, γ, A, 0.0)
    sμ0, sρ0, sE0 = init_shift_sensitivity_ic(mesh, basis, μ0, ρ0, E0)

    μf_pde, ρf_pde, Ef_pde, Σf_pde, sμ_pde, sρ_pde, sE_pde, sΣ_pde =
        run_sensitivity(μ0, ρ0, E0, sμ0, sρ0, sE0, Δt, T, basis, mesh, γ, α;
                        n_iter=n_iter, solver=solver)

    su_pde = (sμ_pde .- u_ad .* sρ_pde) ./ ρf_val
    sp_pde = (γ - 1) .* (sE_pde .- sρ_pde .* u_ad.^2 ./ 2 .- ρf_val .* u_ad .* su_pde)

    # ----------------------------------------------------------
    # Export CSV files
    # ----------------------------------------------------------
    println("    Writing CSVs...")

    xs, _ = flatten_dg(mesh, basis, uf_val)

    # Primal solution
    _, u_s  = flatten_dg(mesh, basis, uf_val)
    _, ρ_s  = flatten_dg(mesh, basis, ρf_val)
    _, p_s  = flatten_dg(mesh, basis, pf_val)
    _, Σ_s  = flatten_dg(mesh, basis, Σf_val)

    open(joinpath(DATA_DIR, "$(prefix)_primal.csv"), "w") do io
        println(io, "x,u,rho,p,sigma")
        for i in eachindex(xs)
            @printf(io, "%.10e,%.10e,%.10e,%.10e,%.10e\n", xs[i], u_s[i], ρ_s[i], p_s[i], Σ_s[i])
        end
    end

    # Helper to write sensitivity CSV
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
    for (label, su_ref, sρ_ref, sp_ref) in [
            ("FD         ", su_fd, sρ_fd, sp_fd),
            ("PDE        ", su_pde, sρ_pde, sp_pde)]
        err_u = norm(su_ref .- su_ad) / max(norm(su_ad), 1e-15)
        err_ρ = norm(sρ_ref .- sρ_ad) / max(norm(sρ_ad), 1e-15)
        err_p = norm(sp_ref .- sp_ad) / max(norm(sp_ad), 1e-15)
        @printf("      %s:  u: %.4e   ρ: %.4e   p: %.4e\n", label, err_u, err_ρ, err_p)
    end

    return nothing
end

# ============================================================
# Run both cases
# ============================================================

params = (A=1.5, p=2, N_e=64, γ=1.4, L=1.0, CFL=0.5,
          α_scale=9.0, n_iter=50, fd_eps=1e-5, solver=:jacobi)

println("="^60)
println("Post-shock case (T = 1.7)")
println("="^60)
generate_case(; prefix="postshock", T=1.7, params...)

println("\nAll data written to: $DATA_DIR")