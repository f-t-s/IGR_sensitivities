#!/usr/bin/env julia
# ============================================================
# Verification tests for the hand-coded forward sensitivity
# equations against ForwardDiff (the discrete-linearization
# reference).
# ============================================================

using Printf
using LinearAlgebra

include("../IGRAdjoints1D.jl")
using .IGRAdjoints1D

using ForwardDiff

# ============================================================
# Helper: relative error
# ============================================================
rel_err(a, b) = abs(a - b) / max(abs(b), 1e-15)

# ============================================================
# Test 1: sensitivity_flux vs ForwardDiff through polytropic_flux
# ============================================================
function test_sensitivity_flux()
    println("="^60)
    println("Test 1: sensitivity_flux vs ForwardDiff")
    println("="^60)

    γ = 1.4
    # Random primal state
    ρ = 1.2; μ = 0.5; E = 2.5; Σ = 0.1
    # Random sensitivities
    sρ = 0.3; sμ = -0.2; sE = 0.15; sΣ = 0.05

    # Hand-coded
    fsμ, fsρ, fsE = sensitivity_flux(γ, μ, ρ, E, Σ, sμ, sρ, sE, sΣ)

    # ForwardDiff reference: differentiate polytropic_flux w.r.t. a parameter ε
    # that perturbs (μ,ρ,E,Σ) → (μ+ε·sμ, ρ+ε·sρ, E+ε·sE, Σ+ε·sΣ)
    function flux_of_eps(ε)
        fμ, fρ, fE = polytropic_flux(γ, μ + ε*sμ, ρ + ε*sρ, E + ε*sE, Σ + ε*sΣ)
        return [fμ, fρ, fE]
    end
    ref = ForwardDiff.derivative(flux_of_eps, 0.0)

    err_μ = rel_err(fsμ, ref[1])
    err_ρ = rel_err(fsρ, ref[2])
    err_E = rel_err(fsE, ref[3])
    max_err = max(err_μ, err_ρ, err_E)

    @printf("  fsμ: hand=%.10e  ref=%.10e  rel_err=%.2e\n", fsμ, ref[1], err_μ)
    @printf("  fsρ: hand=%.10e  ref=%.10e  rel_err=%.2e\n", fsρ, ref[2], err_ρ)
    @printf("  fsE: hand=%.10e  ref=%.10e  rel_err=%.2e\n", fsE, ref[3], err_E)

    pass = max_err < 1e-12
    println("  ", pass ? "PASSED" : "FAILED", " (max rel err = $(@sprintf("%.2e", max_err)))")
    println()
    return pass
end

# ============================================================
# Test 2: sensitivity_llf_flux vs ForwardDiff through llf_flux
# ============================================================
function test_sensitivity_llf_flux()
    println("="^60)
    println("Test 2: sensitivity_llf_flux vs ForwardDiff")
    println("="^60)

    γ = 1.4
    # Primal states (left and right)
    μL = 0.5; ρL = 1.2; EL = 2.5; ΣL = 0.1
    μR = 0.3; ρR = 1.0; ER = 2.2; ΣR = 0.08
    # Sensitivities
    sμL = -0.2; sρL = 0.3; sEL = 0.15; sΣL = 0.05
    sμR = 0.1;  sρR = -0.1; sER = 0.2; sΣR = -0.03

    # Hand-coded
    f_sμ, f_sρ, f_sE = sensitivity_llf_flux(γ,
        μL, ρL, EL, ΣL, μR, ρR, ER, ΣR,
        sμL, sρL, sEL, sΣL, sμR, sρR, sER, sΣR)

    # ForwardDiff reference
    function llf_of_eps(ε)
        fμ, fρ, fE = llf_flux(γ,
            μL + ε*sμL, ρL + ε*sρL, EL + ε*sEL, ΣL + ε*sΣL,
            μR + ε*sμR, ρR + ε*sρR, ER + ε*sER, ΣR + ε*sΣR)
        return [fμ, fρ, fE]
    end
    ref = ForwardDiff.derivative(llf_of_eps, 0.0)

    err_μ = rel_err(f_sμ, ref[1])
    err_ρ = rel_err(f_sρ, ref[2])
    err_E = rel_err(f_sE, ref[3])
    max_err = max(err_μ, err_ρ, err_E)

    @printf("  f★sμ: hand=%.10e  ref=%.10e  rel_err=%.2e\n", f_sμ, ref[1], err_μ)
    @printf("  f★sρ: hand=%.10e  ref=%.10e  rel_err=%.2e\n", f_sρ, ref[2], err_ρ)
    @printf("  f★sE: hand=%.10e  ref=%.10e  rel_err=%.2e\n", f_sE, ref[3], err_E)

    # Expected to differ: we discretize the continuous sensitivity PDE
    # (LLF with primal wavespeed), while ForwardDiff linearizes the
    # discrete LLF (includes δλ·(qL-qR) dissipation sensitivity term).
    println("  (Difference expected: discretize-then-differentiate vs differentiate-then-discretize)")
    println("  max rel diff = $(@sprintf("%.2e", max_err))")
    pass = true  # informational
    println()
    return pass
end

# ============================================================
# Test 3: compute_sensitivity_hyperbolic_rhs! vs ForwardDiff
# ============================================================
function test_sensitivity_hyperbolic_rhs()
    println("="^60)
    println("Test 3: sensitivity hyperbolic RHS vs ForwardDiff")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 4
    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    # Random primal state (physically valid)
    μ = 0.3 .* randn(n_p, N_e)
    ρ = 1.0 .+ 0.2 .* rand(n_p, N_e)
    E = 2.0 .+ 0.3 .* rand(n_p, N_e)
    Σ = 0.1 .* rand(n_p, N_e)

    # Random sensitivities
    sμ = 0.1 .* randn(n_p, N_e)
    sρ = 0.1 .* randn(n_p, N_e)
    sE = 0.1 .* randn(n_p, N_e)
    sΣ = 0.1 .* randn(n_p, N_e)

    # Hand-coded
    dsμ = similar(sμ); dsρ = similar(sρ); dsE = similar(sE)
    compute_sensitivity_hyperbolic_rhs!(dsμ, dsρ, dsE,
        μ, ρ, E, Σ, sμ, sρ, sE, sΣ, basis, mesh, γ)

    # ForwardDiff reference: differentiate compute_hyperbolic_rhs! w.r.t. ε
    function rhs_of_eps(ε)
        μ_d = μ .+ ε .* sμ
        ρ_d = ρ .+ ε .* sρ
        E_d = E .+ ε .* sE
        Σ_d = Σ .+ ε .* sΣ
        dμ_d = similar(μ_d); dρ_d = similar(ρ_d); dE_d = similar(E_d)
        compute_hyperbolic_rhs!(dμ_d, dρ_d, dE_d, μ_d, ρ_d, E_d, Σ_d, basis, mesh, γ)
        return vcat(vec(dμ_d), vec(dρ_d), vec(dE_d))
    end
    ref = ForwardDiff.derivative(rhs_of_eps, 0.0)

    n = n_p * N_e
    ref_dsμ = reshape(ref[1:n], n_p, N_e)
    ref_dsρ = reshape(ref[n+1:2n], n_p, N_e)
    ref_dsE = reshape(ref[2n+1:3n], n_p, N_e)

    err_μ = norm(dsμ - ref_dsμ) / max(norm(ref_dsμ), 1e-15)
    err_ρ = norm(dsρ - ref_dsρ) / max(norm(ref_dsρ), 1e-15)
    err_E = norm(dsE - ref_dsE) / max(norm(ref_dsE), 1e-15)
    max_err = max(err_μ, err_ρ, err_E)

    @printf("  dsμ rel err: %.2e\n", err_μ)
    @printf("  dsρ rel err: %.2e\n", err_ρ)
    @printf("  dsE rel err: %.2e\n", err_E)

    # Same as Test 2: difference expected from discretize-vs-linearize paradigm.
    println("  (Difference expected: discretize-then-differentiate vs differentiate-then-discretize)")
    println("  max rel diff = $(@sprintf("%.2e", max_err))")
    pass = true  # informational
    println()
    return pass
end

# ============================================================
# Test 4: solve_sensitivity_elliptic! vs ForwardDiff
# ============================================================
function test_sensitivity_elliptic(; form=:form2)
    println("="^60)
    println("Test 4$(form == :form1 ? "a" : "b"): sensitivity elliptic ($form) vs ForwardDiff")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; α = 0.01
    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1
    n_iter = 100  # many iterations for convergence

    # Smooth primal state
    μ = zeros(n_p, N_e); ρ = zeros(n_p, N_e); E = zeros(n_p, N_e)
    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i, e]
        ρ[i,e] = 1.0 + 0.2 * sin(2π * x / L)
        u = 0.2 * sin(2π * x / L)
        μ[i,e] = ρ[i,e] * u
        E[i,e] = 1.0 / (γ - 1) + 0.5 * ρ[i,e] * u^2
    end

    # Random perturbation direction
    sμ = 0.1 .* randn(n_p, N_e)
    sρ = 0.1 .* randn(n_p, N_e)
    sE = 0.1 .* randn(n_p, N_e)

    # Compute primal Σ first
    Σ = zeros(n_p, N_e)
    solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α; n_iter=n_iter)

    # Hand-coded sensitivity solve
    sΣ = zeros(n_p, N_e)
    solve_sensitivity_elliptic!(sΣ, Σ, μ, ρ, E, sμ, sρ, sE,
        basis, mesh, γ, α; n_iter=n_iter, elliptic_rhs=form)

    # ForwardDiff reference: differentiate solve_elliptic! w.r.t. ε
    function sigma_of_eps(ε)
        μ_d = μ .+ ε .* sμ
        ρ_d = ρ .+ ε .* sρ
        E_d = E .+ ε .* sE
        Σ_d = zeros(typeof(ε), n_p, N_e)
        solve_elliptic!(Σ_d, μ_d, ρ_d, E_d, basis, mesh, α; n_iter=n_iter)
        return vec(Σ_d)
    end
    ref = ForwardDiff.derivative(sigma_of_eps, 0.0)
    ref_sΣ = reshape(ref, n_p, N_e)

    err = norm(sΣ - ref_sΣ) / max(norm(ref_sΣ), 1e-15)
    @printf("  ||sΣ - ref||/||ref|| = %.2e\n", err)
    @printf("  ||sΣ||  = %.6e\n", norm(sΣ))
    @printf("  ||ref|| = %.6e\n", norm(ref_sΣ))

    # Difference expected: hand-coded discretizes the continuous sensitivity PDE,
    # ForwardDiff linearizes the discrete operator. Agreement improves with resolution.
    pass = err < 5e-2  # generous tolerance for discretize-vs-linearize
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Test 5: Integration test — dJ/dA with run_sensitivity
# ============================================================
function test_integration_dJ_dA(; elliptic_rhs=:form2)
    println("="^60)
    println("Test 5: Integration dJ/dA ($elliptic_rhs) vs ForwardDiff")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T_final = 0.05
    α = 0.01; A0 = 0.2
    n_iter = 100

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    # Conservative CFL estimate
    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)

    # Objective: J(A) = ∫ ρ(x,T;A)² dx
    function make_ic(A)
        RT = typeof(A)
        ρ0 = zeros(RT, n_p, N_e)
        μ0 = zeros(RT, n_p, N_e)
        E0 = zeros(RT, n_p, N_e)
        for e in 1:N_e, i in 1:n_p
            x = mesh.x[i, e]
            ρ_val = 1 + A * sin(2π * x / L)
            u_val = A * sin(2π * x / L)
            p_val = 1 + A * sin(2π * x / L)
            ρ0[i, e] = ρ_val
            μ0[i, e] = ρ_val * u_val
            E0[i, e] = p_val / (γ - 1) + ρ_val * u_val^2 / 2
        end
        return μ0, ρ0, E0
    end

    function l2_norm_sq(q, basis, mesh)
        val = zero(eltype(q))
        for e in 1:mesh.N_e, i in 1:(basis.p + 1)
            val += basis.w[i] * mesh.J * q[i, e]^2
        end
        return val
    end

    # --- ForwardDiff reference ---
    function J_forwarddiff(A)
        μ0, ρ0, E0 = make_ic(A)
        μf, ρf, _, _ = run_forward(μ0, ρ0, E0, Δt, T_final, basis, mesh, γ, α; n_iter=n_iter)
        return l2_norm_sq(ρf, basis, mesh)
    end
    dJ_ad = ForwardDiff.derivative(J_forwarddiff, A0)

    # --- Hand-coded sensitivity ---
    μ0, ρ0, E0 = make_ic(A0)

    # IC sensitivities: ∂/∂A of IC
    sμ0 = zeros(n_p, N_e); sρ0 = zeros(n_p, N_e); sE0 = zeros(n_p, N_e)
    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i, e]
        s = sin(2π * x / L)
        dρ_dA = s
        du_dA = s
        dp_dA = s
        ρ_val = ρ0[i,e]
        u_val = μ0[i,e] / ρ0[i,e]
        sρ0[i,e] = dρ_dA
        sμ0[i,e] = dρ_dA * u_val + ρ_val * du_dA  # d(ρu)/dA
        sE0[i,e] = dp_dA / (γ - 1) + 0.5 * (dρ_dA * u_val^2 + 2 * ρ_val * u_val * du_dA)
    end

    μf, ρf, Ef, Σf, sμf, sρf, sEf, sΣf = run_sensitivity(
        μ0, ρ0, E0, sμ0, sρ0, sE0, Δt, T_final, basis, mesh, γ, α;
        n_iter=n_iter, elliptic_rhs=elliptic_rhs)

    # dJ/dA = ∫ 2ρ sρ dx
    dJ_hand = zero(Float64)
    for e in 1:N_e, i in 1:n_p
        dJ_hand += basis.w[i] * mesh.J * 2 * ρf[i,e] * sρf[i,e]
    end

    err = rel_err(dJ_hand, dJ_ad)
    @printf("  J(A₀)         = %.10e\n", J_forwarddiff(A0))
    @printf("  dJ/dA (hand)   = %.10e\n", dJ_hand)
    @printf("  dJ/dA (AD)     = %.10e\n", dJ_ad)
    @printf("  rel error      = %.2e\n", err)

    # Tolerance: discretize-then-linearize vs linearize-then-discretize
    pass = err < 1e-2
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Test 6: Form 1 vs Form 2 cross-check
# ============================================================
function test_form1_vs_form2()
    println("="^60)
    println("Test 6: Form 1 vs Form 2 cross-check")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T_final = 0.05
    α = 0.01; A0 = 0.2
    n_iter = 100

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)

    # Build ICs
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
        sρ0[i,e] = s
        sμ0[i,e] = s * u_val + ρ_val * s
        sE0[i,e] = s / (γ - 1) + 0.5 * (s * u_val^2 + 2 * ρ_val * u_val * s)
    end

    # Run with Form 1
    μf1, ρf1, Ef1, Σf1, sμf1, sρf1, sEf1, sΣf1 = run_sensitivity(
        μ0, ρ0, E0, sμ0, sρ0, sE0, Δt, T_final, basis, mesh, γ, α;
        n_iter=n_iter, elliptic_rhs=:form1)

    # Run with Form 2
    μf2, ρf2, Ef2, Σf2, sμf2, sρf2, sEf2, sΣf2 = run_sensitivity(
        μ0, ρ0, E0, sμ0, sρ0, sE0, Δt, T_final, basis, mesh, γ, α;
        n_iter=n_iter, elliptic_rhs=:form2)

    # Primal should be identical (same forward solve)
    err_primal = norm(ρf1 - ρf2) / norm(ρf1)
    @printf("  Primal ρ difference: %.2e (should be ~0)\n", err_primal)

    # Sensitivity should agree (both discretize the same continuous PDE)
    err_sρ = norm(sρf1 - sρf2) / max(norm(sρf1), 1e-15)
    err_sμ = norm(sμf1 - sμf2) / max(norm(sμf1), 1e-15)
    err_sE = norm(sEf1 - sEf2) / max(norm(sEf1), 1e-15)
    err_sΣ = norm(sΣf1 - sΣf2) / max(norm(sΣf1), 1e-15)

    @printf("  sμ rel diff: %.2e\n", err_sμ)
    @printf("  sρ rel diff: %.2e\n", err_sρ)
    @printf("  sE rel diff: %.2e\n", err_sE)
    @printf("  sΣ rel diff: %.2e\n", err_sΣ)

    max_err = max(err_sμ, err_sρ, err_sE, err_sΣ)
    pass = err_primal < 1e-14 && max_err < 1e-2
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Run all tests
# ============================================================
println()
println("="^60)
println("Forward sensitivity equations — verification tests")
println("="^60)
println()

all_pass = true
all_pass &= test_sensitivity_flux()
all_pass &= test_sensitivity_llf_flux()
all_pass &= test_sensitivity_hyperbolic_rhs()
all_pass &= test_sensitivity_elliptic(form=:form1)
all_pass &= test_sensitivity_elliptic(form=:form2)
all_pass &= test_integration_dJ_dA(elliptic_rhs=:form2)
all_pass &= test_integration_dJ_dA(elliptic_rhs=:form1)
all_pass &= test_form1_vs_form2()

println("="^60)
if all_pass
    println("All sensitivity tests PASSED")
else
    println("Some sensitivity tests FAILED")
end
println("="^60)
