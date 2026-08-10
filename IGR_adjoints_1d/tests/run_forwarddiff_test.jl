#!/usr/bin/env julia
# ============================================================
# ForwardDiff validation for the IGR Euler DG solver
#
# Tests that ForwardDiff.derivative correctly differentiates
# through the full forward solve (hyperbolic + elliptic),
# validated against central finite differences.
# ============================================================

using Printf

include("../IGRAdjoints1D.jl")
using .IGRAdjoints1D

using ForwardDiff

# ============================================================
# Objective functional: weighted L2 norm squared  ∫ q² dx
# ============================================================
function l2_norm_sq(q, basis, mesh)
    val = zero(eltype(q))
    for e in 1:mesh.N_e, i in 1:(basis.p + 1)
        val += basis.w[i] * mesh.J * q[i, e]^2
    end
    return val
end

# ============================================================
# Test 1: dJ/dα (sensitivity to IGR parameter)
# ============================================================
function test_dJ_dalpha()
    println("="^60)
    println("Test 1: dJ/dα  (IGR parameter sensitivity)")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T_final = 0.1
    α0 = 0.01

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    # Acoustic pulse IC (has velocity gradients → Σ ≠ 0)
    ε = 0.2
    μ0 = zeros(n_p, N_e)
    ρ0 = zeros(n_p, N_e)
    E0 = zeros(n_p, N_e)
    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i, e]
        ρ_val = 1.0 + ε * sin(2π * x / L)
        u_val = ε * sin(2π * x / L)
        p_val = 1.0 + ε * sin(2π * x / L)
        ρ0[i, e] = ρ_val
        μ0[i, e] = ρ_val * u_val
        E0[i, e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
    end

    # CFL time step
    max_ws = 0.0
    for e in 1:N_e, i in 1:n_p
        ws = max_wavespeed(γ, μ0[i,e], ρ0[i,e], E0[i,e])
        max_ws = max(max_ws, ws)
    end
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws)

    # Objective: J(α) = ∫ ρ(x,T;α)² dx
    function J(α)
        μf, ρf, _, _ = run_forward(μ0, ρ0, E0, Δt, T_final, basis, mesh, γ, α; n_iter=100)
        return l2_norm_sq(ρf, basis, mesh)
    end

    # ForwardDiff
    dJ_ad = ForwardDiff.derivative(J, α0)

    # Central finite differences
    h = 1e-7
    dJ_fd = (J(α0 + h) - J(α0 - h)) / (2h)

    rel_err = abs(dJ_ad - dJ_fd) / max(abs(dJ_fd), 1e-15)

    @printf("  J(α₀)      = %.10e\n", J(α0))
    @printf("  dJ/dα (AD)  = %.10e\n", dJ_ad)
    @printf("  dJ/dα (FD)  = %.10e\n", dJ_fd)
    @printf("  rel error   = %.2e\n", rel_err)

    pass = rel_err < 1e-5
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Test 2: dJ/dA (IC amplitude sensitivity, with IGR)
# ============================================================
function test_dJ_dA_with_igr()
    println("="^60)
    println("Test 2: dJ/dA  (IC amplitude, α > 0)")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T_final = 0.1
    α = 0.01; A0 = 0.2

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    # Conservative CFL estimate
    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)

    # Objective: J(A) = ∫ ρ(x,T;A)² dx
    # ICs constructed inside so A flows through as a Dual
    function J(A)
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
        μf, ρf, _, _ = run_forward(μ0, ρ0, E0, Δt, T_final, basis, mesh, γ, α; n_iter=100)
        return l2_norm_sq(ρf, basis, mesh)
    end

    dJ_ad = ForwardDiff.derivative(J, A0)

    h = 1e-7
    dJ_fd = (J(A0 + h) - J(A0 - h)) / (2h)

    rel_err = abs(dJ_ad - dJ_fd) / max(abs(dJ_fd), 1e-15)

    @printf("  J(A₀)      = %.10e\n", J(A0))
    @printf("  dJ/dA (AD)  = %.10e\n", dJ_ad)
    @printf("  dJ/dA (FD)  = %.10e\n", dJ_fd)
    @printf("  rel error   = %.2e\n", rel_err)

    pass = rel_err < 1e-5
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Test 3: dJ/dA (IC amplitude sensitivity, pure Euler, α = 0)
# ============================================================
function test_dJ_dA_no_igr()
    println("="^60)
    println("Test 3: dJ/dA  (IC amplitude, α = 0, pure Euler)")
    println("="^60)

    γ = 1.4; L = 1.0; p = 3; N_e = 8; CFL = 0.3; T_final = 0.1
    α = 0.0; A0 = 0.2

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    max_ws_est = 2.5
    Δt = CFL * mesh.Δx / ((2*p + 1) * max_ws_est)

    function J(A)
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
        μf, ρf, _, _ = run_forward(μ0, ρ0, E0, Δt, T_final, basis, mesh, γ, α; n_iter=100)
        return l2_norm_sq(ρf, basis, mesh)
    end

    dJ_ad = ForwardDiff.derivative(J, A0)

    h = 1e-7
    dJ_fd = (J(A0 + h) - J(A0 - h)) / (2h)

    rel_err = abs(dJ_ad - dJ_fd) / max(abs(dJ_fd), 1e-15)

    @printf("  J(A₀)      = %.10e\n", J(A0))
    @printf("  dJ/dA (AD)  = %.10e\n", dJ_ad)
    @printf("  dJ/dA (FD)  = %.10e\n", dJ_fd)
    @printf("  rel error   = %.2e\n", rel_err)

    pass = rel_err < 1e-5
    println("  ", pass ? "PASSED" : "FAILED")
    println()
    return pass
end

# ============================================================
# Run all tests
# ============================================================
println()
println("="^60)
println("ForwardDiff validation for IGR Euler DG solver")
println("="^60)
println()

all_pass = true
all_pass &= test_dJ_dalpha()
all_pass &= test_dJ_dA_with_igr()
all_pass &= test_dJ_dA_no_igr()

println("="^60)
if all_pass
    println("All ForwardDiff tests PASSED")
else
    println("Some ForwardDiff tests FAILED")
end
println("="^60)
