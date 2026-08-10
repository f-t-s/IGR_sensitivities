#!/usr/bin/env julia
# ============================================================
# Test driver for the 1D periodic DG Euler solver
# ============================================================

using Printf

# Load the module
include("../IGRAdjoints1D.jl")
using .IGRAdjoints1D

# ============================================================
# Test 1: Basis validation
# ============================================================
function test_basis()
    println("="^60)
    println("Test 1: Basis validation")
    println("="^60)

    for p in 1:5
        basis = DGBasis(p)

        # Check quadrature weights sum to 2 (integral of 1 over [-1,1])
        w_sum = sum(basis.w)
        @assert abs(w_sum - 2.0) < 1e-14 "Weights don't sum to 2 for p=$p: $w_sum"

        # Check D is exact for monomials up to degree p
        max_err = 0.0
        for k in 1:p
            f = basis.ξ .^ k
            df_exact = k .* basis.ξ .^ (k - 1)
            df_approx = basis.D * f
            err = maximum(abs.(df_approx - df_exact))
            max_err = max(max_err, err)
        end
        @printf("  p=%d: weight sum error = %.2e, D matrix error (deg ≤ p) = %.2e\n",
                p, abs(w_sum - 2.0), max_err)
        @assert max_err < 1e-12 "D matrix not exact for p=$p"
    end
    println("  PASSED\n")
end

# ============================================================
# Test 2: Free-stream preservation
# ============================================================
function test_freestream()
    println("="^60)
    println("Test 2: Free-stream preservation")
    println("="^60)

    γ = 1.4
    for p in 1:4
        basis = DGBasis(p)
        mesh = PeriodicMesh1D(8, 1.0, basis)

        μ, ρ, E = init_constant_state(mesh, γ)
        Σ = zeros(size(μ))

        dμ = similar(μ); dρ = similar(ρ); dE = similar(E)
        compute_hyperbolic_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ)

        err_μ = maximum(abs.(dμ))
        err_ρ = maximum(abs.(dρ))
        err_E = maximum(abs.(dE))
        @printf("  p=%d: max|dμ| = %.2e, max|dρ| = %.2e, max|dE| = %.2e\n",
                p, err_μ, err_ρ, err_E)
        @assert err_μ < 1e-12 "Free-stream not preserved for μ, p=$p"
        @assert err_ρ < 1e-12 "Free-stream not preserved for ρ, p=$p"
        @assert err_E < 1e-12 "Free-stream not preserved for E, p=$p"
    end
    println("  PASSED\n")
end

# ============================================================
# Test 3: Entropy wave convergence (order p+1)
# ============================================================
function test_entropy_wave_convergence()
    println("="^60)
    println("Test 3: Entropy wave convergence")
    println("="^60)

    γ = 1.4
    L = 1.0
    T = 1.0  # one full period

    for p in 1:4
        println("  p = $p:")
        errors = Float64[]
        N_es = [8, 16, 32, 64]

        for N_e in N_es
            basis = DGBasis(p)
            mesh = PeriodicMesh1D(N_e, L, basis)

            μ0, ρ0, E0 = init_entropy_wave(mesh, γ)

            # CFL-based time step
            max_ws = 0.0
            for e in 1:N_e
                for i in 1:(p+1)
                    ws = max_wavespeed(γ, μ0[i,e], ρ0[i,e], E0[i,e])
                    max_ws = max(max_ws, ws)
                end
            end
            Δt = 0.5 * mesh.Δx / ((2*p + 1) * max_ws)

            # Forward solve with α=0 (pure Euler, no IGR)
            μf, ρf, Ef, _ = run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, 0.0; n_iter=0)

            err_ρ = l2_error(ρf, ρ0, basis, mesh)
            push!(errors, err_ρ)
            @printf("    N_e=%3d, Δx=%.4f, L2(ρ) = %.4e\n", N_e, mesh.Δx, err_ρ)
        end

        # Compute convergence orders
        for i in 2:length(errors)
            order = log(errors[i-1] / errors[i]) / log(2)
            @printf("    Order (N_e %d→%d): %.2f\n", N_es[i-1], N_es[i], order)
        end
        println()
    end
end

# ============================================================
# Test 4: SIP elliptic solver (manufactured solution)
# ============================================================
function test_elliptic_manufactured()
    println("="^60)
    println("Test 4: SIP elliptic solver (manufactured solution)")
    println("="^60)

    # Manufactured solution: Σ_exact = sin(2πx), ρ = 1 (constant)
    # Equation: Σ/ρ - α ∂²(Σ/ρ)/∂x² = Σ + α(2π)²Σ = (1 + 4π²α) sin(2πx)
    α = 0.1
    L = 1.0

    for p in 1:4
        println("  p = $p:")
        errors = Float64[]
        N_es = [4, 8, 16, 32]

        for N_e in N_es
            basis = DGBasis(p)
            mesh = PeriodicMesh1D(N_e, L, basis)
            n_p = p + 1

            ρ = ones(n_p, N_e)
            Σ = zeros(n_p, N_e)

            # Exact solution and RHS
            Σ_exact = similar(Σ)
            b = similar(Σ)
            for e in 1:N_e
                for i in 1:n_p
                    Σ_exact[i,e] = sin(2π * mesh.x[i,e] / L)
                    b[i,e] = basis.w[i] * mesh.J * (1 + 4π^2 * α / L^2) * Σ_exact[i,e]
                end
            end

            apply_A!(y, x) = apply_sip!(y, x, ρ, basis, mesh, α)
            n_iter = cg_solve!(Σ, apply_A!, b, 1e-13, 1000)

            err = l2_error(Σ, Σ_exact, basis, mesh)
            push!(errors, err)
            @printf("    N_e=%3d, L2 error = %.4e, CG iters = %d\n", N_e, err, n_iter)
        end

        for i in 2:length(errors)
            if errors[i] > 1e-14
                order = log(errors[i-1] / errors[i]) / log(2)
                @printf("    Order (N_e %d→%d): %.2f\n", N_es[i-1], N_es[i], order)
            end
        end
        println()
    end
end

# ============================================================
# Test 5: IGR Euler — acoustic pulse with α > 0
# ============================================================
function test_igr_acoustic_pulse()
    println("="^60)
    println("Test 5: IGR Euler — acoustic pulse with α > 0")
    println("="^60)

    # Acoustic pulse: ρ = 1 + ε sin(2πx), u = ε sin(2πx), p = 1 + ε sin(2πx)
    # The velocity perturbation ensures ∂u/∂x ≠ 0, so R ≠ 0 and Σ ≠ 0.
    γ = 1.4
    L = 1.0
    T = 0.5
    p = 3
    N_e = 16
    ε = 0.1

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    μ0 = similar(mesh.x)
    ρ0 = similar(mesh.x)
    E0 = similar(mesh.x)
    for e in 1:N_e
        for i in 1:n_p
            x = mesh.x[i,e]
            ρ_val = 1.0 + ε * sin(2π * x / L)
            u_val = ε * sin(2π * x / L)
            p_val = 1.0 + ε * sin(2π * x / L)
            ρ0[i,e] = ρ_val
            μ0[i,e] = ρ_val * u_val
            E0[i,e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
    end

    # CFL-based time step
    max_ws = 0.0
    for e in 1:N_e
        for i in 1:n_p
            ws = max_wavespeed(γ, μ0[i,e], ρ0[i,e], E0[i,e])
            max_ws = max(max_ws, ws)
        end
    end
    Δt = 0.5 * mesh.Δx / ((2*p + 1) * max_ws)

    # Run without IGR (reference)
    μf0, ρf0, _, _ = run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, 0.0)

    # Run with several α values
    αs = [1e-6, 1e-4, 1e-2]
    for α in αs
        μf, ρf, Ef, Σf = run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=100)

        # Compare to α=0 reference
        diff_ρ = l2_error(ρf, ρf0, basis, mesh)
        max_Σ = maximum(abs.(Σf))
        @printf("  α=%.0e: L2(ρ - ρ_ref) = %.4e, max|Σ| = %.4e\n", α, diff_ρ, max_Σ)

        @assert isfinite(diff_ρ) "Solution blew up for α=$α"
        @assert max_Σ > 0 "Σ should be non-zero for non-uniform velocity"
    end
    println("  PASSED\n")
end

# ============================================================
# Test 6: IGR Euler — pressure jump (Sod-like on periodic domain)
# ============================================================
function test_igr_pressure_jump()
    println("="^60)
    println("Test 6: IGR Euler — pressure jump with α > 0")
    println("="^60)

    # Smoothed pressure jump creates velocity gradients via wave interaction
    γ = 1.4
    L = 1.0
    T = 0.1   # short time to avoid strong shocks
    p = 2
    N_e = 32
    α = 1e-3

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    # Smooth pressure perturbation: ρ = 1, u = 0, p = 1 + 0.5 sin(2πx)
    # Zero velocity but non-zero pressure gradient → generates velocity → non-zero Σ
    μ0 = similar(mesh.x)
    ρ0 = similar(mesh.x)
    E0 = similar(mesh.x)
    for e in 1:N_e
        for i in 1:n_p
            x = mesh.x[i,e]
            ρ_val = 1.0
            u_val = 0.0
            p_val = 1.0 + 0.5 * sin(2π * x / L)
            ρ0[i,e] = ρ_val
            μ0[i,e] = ρ_val * u_val
            E0[i,e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
    end

    # CFL-based time step (conservative)
    max_ws = 0.0
    for e in 1:N_e
        for i in 1:n_p
            ws = max_wavespeed(γ, μ0[i,e], ρ0[i,e], E0[i,e])
            max_ws = max(max_ws, ws)
        end
    end
    Δt = 0.3 * mesh.Δx / ((2*p + 1) * max_ws)

    μf, ρf, Ef, Σf = run_forward(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=100)

    max_ρ = maximum(ρf)
    min_ρ = minimum(ρf)
    max_Σ = maximum(abs.(Σf))
    @printf("  ρ range: [%.4f, %.4f], max|Σ| = %.4e\n", min_ρ, max_ρ, max_Σ)
    @assert isfinite(max_ρ) "Solution blew up"
    @assert min_ρ > 0.0 "Negative density"
    @assert max_Σ > 0 "Σ should be non-zero after wave interaction"
    println("  PASSED\n")
end

# ============================================================
# Run all tests
# ============================================================
test_basis()
test_freestream()
test_entropy_wave_convergence()
test_elliptic_manufactured()
test_igr_acoustic_pulse()
test_igr_pressure_jump()

println("All tests completed.")
