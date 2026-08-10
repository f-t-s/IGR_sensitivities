#!/usr/bin/env julia
# ============================================================
# Verification tests for Enzyme reverse-mode AD
# against ForwardDiff and finite differences.
# ============================================================

using Printf
using LinearAlgebra

include("../IGRAdjoints1D.jl")
using .IGRAdjoints1D

using Enzyme
using ForwardDiff

# ============================================================
# Helper
# ============================================================
rel_err(a, b) = abs(a - b) / max(abs(b), 1e-15)

# ============================================================
# Test 1: dJ/dA — Enzyme vs ForwardDiff vs FD
# ============================================================
function test_dJ_dA()
    println("="^60)
    println("Test 1: dJ/dA — Enzyme vs ForwardDiff vs FD")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T = 0.05
    α = 0.01; A0 = 0.2; n_iter = 100

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)
    n_steps = ceil(Int, T / Δt)
    Δt_adj = T / n_steps

    # --- ForwardDiff reference ---
    function J_of_A_fd(A)
        μ0 = zeros(typeof(A), n_p, N_e)
        ρ0 = zeros(typeof(A), n_p, N_e)
        E0 = zeros(typeof(A), n_p, N_e)
        for e in 1:N_e, i in 1:n_p
            x = mesh.x[i, e]
            s = sin(2π * x / L)
            ρ_val = 1 + A * s
            u_val = A * s
            p_val = 1 + A * s
            ρ0[i,e] = ρ_val
            μ0[i,e] = ρ_val * u_val
            E0[i,e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
        μf, ρf, _, _ = run_forward(μ0, ρ0, E0, Δt_adj, T, basis, mesh, γ, α; n_iter=n_iter)
        val = zero(eltype(ρf))
        for e in 1:N_e, i in 1:n_p
            val += basis.w[i] * mesh.J * ρf[i,e]^2
        end
        return val
    end
    dJ_forwarddiff = ForwardDiff.derivative(J_of_A_fd, A0)
    J_val_ref = J_of_A_fd(A0)

    # --- Finite differences ---
    ε = 1e-7
    dJ_fd = (J_of_A_fd(A0 + ε) - J_of_A_fd(A0 - ε)) / (2ε)

    # --- Enzyme ---
    println("  Running Enzyme...")
    J_val_enz, dJ_enzyme = enzyme_dJ_dA(A0, p, N_e, L, γ, α, CFL, T; n_iter=n_iter)

    err_enz_ad = rel_err(dJ_enzyme, dJ_forwarddiff)
    err_enz_fd = rel_err(dJ_enzyme, dJ_fd)
    err_ad_fd  = rel_err(dJ_forwarddiff, dJ_fd)

    @printf("  J(A₀)             = %.10e\n", J_val_ref)
    @printf("  dJ/dA (Enzyme)     = %.10e\n", dJ_enzyme)
    @printf("  dJ/dA (ForwardDiff)= %.10e\n", dJ_forwarddiff)
    @printf("  dJ/dA (FD)         = %.10e\n", dJ_fd)
    @printf("  |Enzyme - FwdDiff| / |FwdDiff| = %.2e\n", err_enz_ad)
    @printf("  |Enzyme - FD|      / |FD|      = %.2e\n", err_enz_fd)
    @printf("  |FwdDiff - FD|     / |FD|      = %.2e\n", err_ad_fd)

    # Enzyme and ForwardDiff should match to near machine precision
    pass = err_enz_ad < 1e-8
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Test 2: Forward solve consistency — run_forward vs _enzyme_objective
# ============================================================
function test_forward_consistency()
    println("="^60)
    println("Test 2: Forward solve consistency")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T = 0.05
    α = 0.01; A0 = 0.2; n_iter = 100

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)
    n_steps = ceil(Int, T / Δt)
    Δt_adj = T / n_steps

    # Build IC
    μ0 = zeros(n_p, N_e); ρ0 = zeros(n_p, N_e); E0 = zeros(n_p, N_e)
    build_acoustic_pulse_ic!(μ0, ρ0, E0, A0, mesh.x, L, γ)

    # --- run_forward (canonical forward solve) ---
    μf, ρf, _, _ = run_forward(μ0, ρ0, E0, Δt_adj, T, basis, mesh, γ, α; n_iter=n_iter)
    J_forward = 0.0
    for e in 1:N_e, i in 1:n_p
        J_forward += basis.w[i] * mesh.J * ρf[i,e]^2
    end

    # --- _enzyme_objective (inlined SSP-RK3, called directly) ---
    J_enzyme_obj = IGRAdjoints1D._enzyme_objective(
        α, μ0, ρ0, E0, n_p, N_e, n_steps, Δt_adj, basis, mesh, γ, n_iter, :pcg)

    # --- Enzyme autodiff (primal value from reverse pass) ---
    J_enzyme_ad, _ = enzyme_dJ_dA(A0, p, N_e, L, γ, α, CFL, T; n_iter=n_iter)

    err_obj = rel_err(J_enzyme_obj, J_forward)
    err_ad  = rel_err(J_enzyme_ad, J_forward)

    @printf("  J (run_forward)       = %.15e\n", J_forward)
    @printf("  J (_enzyme_objective) = %.15e\n", J_enzyme_obj)
    @printf("  J (Enzyme autodiff)   = %.15e\n", J_enzyme_ad)
    @printf("  |obj - forward| / |forward| = %.2e\n", err_obj)
    @printf("  |ad  - forward| / |forward| = %.2e\n", err_ad)

    pass = err_obj < 1e-14 && err_ad < 1e-14
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Test 3: Enzyme vs PDE-based forward sensitivity
# ============================================================
function test_enzyme_vs_pde_sensitivity()
    println("="^60)
    println("Test 3: Enzyme dJ/dA vs PDE-based sensitivity dJ/dA")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T = 0.05
    α = 0.01; A0 = 0.2; n_iter = 100

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)
    n_steps = ceil(Int, T / Δt)
    Δt_adj = T / n_steps

    # --- Enzyme ---
    println("  Running Enzyme...")
    _, dJ_enzyme = enzyme_dJ_dA(A0, p, N_e, L, γ, α, CFL, T; n_iter=n_iter)

    # --- PDE-based sensitivity ---
    println("  Running PDE sensitivity...")
    μ0 = zeros(n_p, N_e); ρ0 = zeros(n_p, N_e); E0 = zeros(n_p, N_e)
    sμ0 = zeros(n_p, N_e); sρ0 = zeros(n_p, N_e); sE0 = zeros(n_p, N_e)
    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i, e]
        s = sin(2π * x / L)
        ρ_val = 1 + A0 * s
        u_val = A0 * s
        p_val = 1 + A0 * s
        ρ0[i,e] = ρ_val
        μ0[i,e] = ρ_val * u_val
        E0[i,e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        # IC sensitivities: ∂/∂A
        dρ_dA = s
        du_dA = s
        dp_dA = s
        sρ0[i,e] = dρ_dA
        sμ0[i,e] = dρ_dA * u_val + ρ_val * du_dA
        sE0[i,e] = dp_dA / (γ - 1) + 0.5 * (dρ_dA * u_val^2 + 2 * ρ_val * u_val * du_dA)
    end

    μf, ρf, Ef, Σf, sμf, sρf, sEf, sΣf = run_sensitivity(
        μ0, ρ0, E0, sμ0, sρ0, sE0, Δt_adj, T, basis, mesh, γ, α;
        n_iter=n_iter, elliptic_rhs=:form2)

    # dJ/dA = ∫ 2ρ sρ dx
    dJ_pde = zero(Float64)
    for e in 1:N_e, i in 1:n_p
        dJ_pde += basis.w[i] * mesh.J * 2 * ρf[i,e] * sρf[i,e]
    end

    err = rel_err(dJ_pde, dJ_enzyme)
    @printf("  dJ/dA (Enzyme) = %.10e\n", dJ_enzyme)
    @printf("  dJ/dA (PDE)    = %.10e\n", dJ_pde)
    @printf("  rel diff       = %.2e\n", err)
    println("  (Difference expected: discretize-then-differentiate vs differentiate-then-discretize)")

    # Generous tolerance — the two approaches solve different discrete problems
    pass = err < 1e-1
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Run all tests
# ============================================================
println()
println("="^60)
println("Enzyme reverse-mode AD — verification tests")
println("="^60)
println()

all_pass = true
all_pass &= test_dJ_dA()
all_pass &= test_forward_consistency()
all_pass &= test_enzyme_vs_pde_sensitivity()

println("="^60)
if all_pass
    println("All Enzyme tests PASSED")
else
    println("Some Enzyme tests FAILED")
end
println("="^60)
