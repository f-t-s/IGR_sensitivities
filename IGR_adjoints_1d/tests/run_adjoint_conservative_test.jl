#!/usr/bin/env julia
# ============================================================
# Verification tests for the conservative PDE-based adjoint solver
# (Π approach) against Enzyme reverse-mode AD and finite differences.
#
# Compares:
#   1. Scalar gradient dJ/dA: conservative adjoint vs Enzyme vs FD
#   2. Field-level adjoint: conservative q†(0) vs Enzyme ∂J/∂q₀
#      (accounting for mass-matrix scaling: dJ/dq₀ = w·J · q†)
#   3. Multiple IC types and α values
# ============================================================

using Printf
using LinearAlgebra

include("../IGRAdjoints1D.jl")
using .IGRAdjoints1D

rel_err(a, b) = abs(a - b) / max(abs(b), 1e-15)

# Helper: compute dJ/dA from adjoint fields (conservative variables)
function dJ_dA_from_adjoint(aμ, aρ, aE, A0, basis, mesh, γ, L, ic_type)
    n_p = basis.p + 1
    N_e = mesh.N_e
    dJ = zero(Float64)
    wρ, wu, wp = ic_dA_weights(ic_type)
    for e in 1:N_e, i in 1:n_p
        x = mesh.x[i,e]; s = sin(2π * x / L)

        # Reconstruct IC primitives to get ∂(conservative)/∂A
        μ0_tmp = zeros(1,1); ρ0_tmp = zeros(1,1); E0_tmp = zeros(1,1)
        builder! = IC_BUILDERS[ic_type][1]
        mesh_x_tmp = fill(x, 1, 1)
        builder!(μ0_tmp, ρ0_tmp, E0_tmp, A0, mesh_x_tmp, L, γ)

        ρ_val = ρ0_tmp[1,1]
        u_val = μ0_tmp[1,1] / ρ_val

        # ∂q₀/∂A via chain rule from primitives (ρ,u,p) → (μ,ρ,E)
        dρ_dA = wρ * s
        du_dA = wu * s
        dp_dA = wp * s
        dμ_dA = ρ_val * du_dA + u_val * dρ_dA
        dE_dA = dp_dA / (γ - 1) + 0.5 * (dρ_dA * u_val^2 + 2 * ρ_val * u_val * du_dA)

        dJ += basis.w[i] * mesh.J * (aμ[i,e] * dμ_dA + aρ[i,e] * dρ_dA + aE[i,e] * dE_dA)
    end
    return dJ
end

# ============================================================
# Core test: compare conservative adjoint vs Enzyme vs FD
# ============================================================
function test_adjoint(; p=2, N_e=16, CFL=0.3, T=0.05, α=0.0,
                       n_iter=10, A0=0.2, ic_type=:acoustic_pulse,
                       tol=0.01, label="")
    γ = 1.4; L = 1.0

    println("\n" * "="^70)
    if isempty(label)
        @printf("Test: p=%d, N_e=%d, α=%.4f, T=%.2f, ic=%s\n", p, N_e, α, T, ic_type)
    else
        println(label)
        @printf("  p=%d, N_e=%d, α=%.4f, T=%.2f, ic=%s\n", p, N_e, α, T, ic_type)
    end
    println("="^70)

    basis = DGBasis(p)
    mesh = PeriodicMesh1D(N_e, L, basis)
    n_p = p + 1

    Δt = CFL * mesh.Δx / ((2*p + 1) * 2.5)
    n_steps_est = ceil(Int, T / Δt)
    Δt = T / n_steps_est

    builder! = IC_BUILDERS[ic_type][1]

    # --- Enzyme ---
    _, adj_ρ_enz, adj_u_enz, adj_p_enz = enzyme_adjoint_ic(
        A0, p, N_e, L, γ, α, CFL, T; n_iter=n_iter, ic_type=ic_type)
    wρ, wu, wp = ic_dA_weights(ic_type)
    dJ_enzyme = sum(
        (wρ * adj_ρ_enz[i,e] + wu * adj_u_enz[i,e] + wp * adj_p_enz[i,e]) *
        sin(2π * mesh.x[i,e] / L) for e in 1:N_e for i in 1:n_p)

    # --- Forward solve ---
    μ0 = zeros(n_p, N_e); ρ0 = zeros(n_p, N_e); E0 = zeros(n_p, N_e)
    builder!(μ0, ρ0, E0, A0, mesh.x, L, γ)
    snapshots, n_steps = run_forward_store(μ0, ρ0, E0, Δt, T, basis, mesh, γ, α; n_iter=n_iter)

    # --- Conservative adjoint ---
    aμ, aρ, aE = run_adjoint_conservative(snapshots, n_steps, Δt, T, basis, mesh, γ, α;
                                            n_iter=n_iter, objective=:l2_density)
    dJ_con = dJ_dA_from_adjoint(aμ, aρ, aE, A0, basis, mesh, γ, L, ic_type)

    # --- FD reference ---
    function J_of_A(A)
        μ0l = zeros(n_p, N_e); ρ0l = zeros(n_p, N_e); E0l = zeros(n_p, N_e)
        builder!(μ0l, ρ0l, E0l, A, mesh.x, L, γ)
        _, ρf, _, _ = run_forward(μ0l, ρ0l, E0l, Δt, T, basis, mesh, γ, α; n_iter=n_iter)
        sum(basis.w[i] * mesh.J * ρf[i,e]^2 for e in 1:N_e for i in 1:n_p)
    end
    dJ_fd = (J_of_A(A0 + 1e-6) - J_of_A(A0 - 1e-6)) / 2e-6

    # --- Scalar dJ/dA ---
    @printf("  Scalar dJ/dA:\n")
    @printf("    FD:            %.8e\n", dJ_fd)
    @printf("    Enzyme:        %.8e\n", dJ_enzyme)
    @printf("    Conservative:  %.8e\n", dJ_con)
    @printf("    Enzyme vs FD:  %.4e\n", rel_err(dJ_enzyme, dJ_fd))
    @printf("    Con vs FD:     %.4e\n", rel_err(dJ_con, dJ_fd))

    # --- Field-level: transform to primitive and scale by mass matrix ---
    u0 = μ0 ./ ρ0
    adj_ρ_prim = aρ .+ u0 .* aμ .+ (u0.^2 ./ 2) .* aE
    adj_u_prim = ρ0 .* aμ .+ ρ0 .* u0 .* aE
    adj_p_prim = aE ./ (γ - 1)
    adj_ρ_s = similar(adj_ρ_prim)
    adj_u_s = similar(adj_u_prim)
    adj_p_s = similar(adj_p_prim)
    for e in 1:N_e, i in 1:n_p
        wJ = basis.w[i] * mesh.J
        adj_ρ_s[i,e] = wJ * adj_ρ_prim[i,e]
        adj_u_s[i,e] = wJ * adj_u_prim[i,e]
        adj_p_s[i,e] = wJ * adj_p_prim[i,e]
    end
    err_ρ = norm(adj_ρ_s - adj_ρ_enz) / max(norm(adj_ρ_enz), 1e-15)
    err_u = norm(adj_u_s - adj_u_enz) / max(norm(adj_u_enz), 1e-15)
    err_p = norm(adj_p_s - adj_p_enz) / max(norm(adj_p_enz), 1e-15)
    @printf("\n  Conservative vs Enzyme fields (w·J·q†_prim, relative L2):\n")
    @printf("    adj_ρ: %.4e\n", err_ρ)
    @printf("    adj_u: %.4e\n", err_u)
    @printf("    adj_p: %.4e\n", err_p)

    pass = rel_err(dJ_con, dJ_fd) < tol
    println(pass ? "  PASS" : "  FAIL")
    return pass
end

# ============================================================
# Sweep test: run over a list of parameter values
# ============================================================
function test_sweep(; sweep_param::Symbol, sweep_values,
                      p=2, N_e=16, CFL=0.3, T=0.05, α=0.0,
                      n_iter=10, A0=0.2, ic_type=:acoustic_pulse,
                      tol=0.01, label="")
    γ = 1.4; L = 1.0

    println("\n" * "="^70)
    if !isempty(label)
        println(label)
    else
        @printf("Sweep over %s\n", sweep_param)
    end
    println("="^70)
    @printf("  %-18s  %14s  %14s  %14s  %10s  %s\n",
            String(sweep_param), "FD", "Enzyme", "Con", "err_con", "")
    println("  " * "-"^85)

    all_pass = true
    for val in sweep_values
        _p = p; _N_e = N_e; _T = T; _α = α; _n_iter = n_iter
        _ic_type = ic_type
        if sweep_param == :α;       _α = val
        elseif sweep_param == :T;   _T = val
        elseif sweep_param == :N_e; _N_e = val
        elseif sweep_param == :p;   _p = val
        elseif sweep_param == :ic_type; _ic_type = val
        end

        basis = DGBasis(_p)
        mesh = PeriodicMesh1D(_N_e, L, basis)
        n_p = _p + 1

        Δt = CFL * mesh.Δx / ((2*_p + 1) * 2.5)
        n_steps_est = ceil(Int, _T / Δt)
        Δt = _T / n_steps_est

        builder! = IC_BUILDERS[_ic_type][1]

        # Enzyme
        _, dJ_enzyme = enzyme_dJ_dA(A0, _p, _N_e, L, γ, _α, CFL, _T; n_iter=_n_iter)

        # Forward + conservative adjoint
        μ0 = zeros(n_p, _N_e); ρ0 = zeros(n_p, _N_e); E0 = zeros(n_p, _N_e)
        builder!(μ0, ρ0, E0, A0, mesh.x, L, γ)
        snapshots, n_steps = run_forward_store(μ0, ρ0, E0, Δt, _T, basis, mesh, γ, _α; n_iter=_n_iter)

        aμ, aρ, aE = run_adjoint_conservative(snapshots, n_steps, Δt, _T, basis, mesh, γ, _α;
                                                n_iter=_n_iter, objective=:l2_density)
        dJ_con = dJ_dA_from_adjoint(aμ, aρ, aE, A0, basis, mesh, γ, L, _ic_type)

        # FD
        function J_of_A(A)
            μ0l = zeros(n_p, _N_e); ρ0l = zeros(n_p, _N_e); E0l = zeros(n_p, _N_e)
            builder!(μ0l, ρ0l, E0l, A, mesh.x, L, γ)
            _, ρf, _, _ = run_forward(μ0l, ρ0l, E0l, Δt, _T, basis, mesh, γ, _α; n_iter=_n_iter)
            sum(basis.w[i] * mesh.J * ρf[i,e]^2 for e in 1:_N_e for i in 1:n_p)
        end
        dJ_fd = (J_of_A(A0 + 1e-6) - J_of_A(A0 - 1e-6)) / 2e-6

        err_con = rel_err(dJ_con, dJ_fd)
        pass = err_con < tol
        all_pass &= pass

        val_str = sweep_param == :ic_type ? String(val) : @sprintf("%.4g", val)
        @printf("  %-18s  %+.6e  %+.6e  %+.6e  %.2e  %s\n",
                val_str, dJ_fd, dJ_enzyme, dJ_con, err_con,
                pass ? "PASS" : "FAIL")
    end

    println(all_pass ? "  All PASS" : "  Some FAILED")
    return all_pass
end

# ============================================================
# Run all tests
# ============================================================
println("Conservative adjoint PDE (Π approach) — verification")
println("="^70)

# Test 1: α=0, conservative vs Enzyme (should match DtO exactly)
pass1 = test_adjoint(α=0.0, n_iter=10, tol=0.01,
    label="Test 1: Conservative vs Enzyme — α=0")

# Test 2: α>0, conservative vs Enzyme
pass2 = test_adjoint(α=0.01, n_iter=200, tol=0.05,
    label="Test 2: Conservative vs Enzyme — α>0")

# Test 3: IC type sweep at α=0
pass3 = test_sweep(sweep_param=:ic_type,
    sweep_values=[:acoustic_pulse, :entropy_wave, :density_pulse,
                  :pressure_pulse, :velocity_pulse],
    α=0.0, n_iter=10, tol=0.01,
    label="Test 3: IC type sweep — α=0")

# Test 4: α sweep
pass4 = test_sweep(sweep_param=:α,
    sweep_values=[0.0, 0.001, 0.01, 0.05],
    n_iter=200, tol=0.05,
    label="Test 4: α sweep — conservative vs FD vs Enzyme")

# Test 5: Longer time
pass5 = test_adjoint(α=0.01, n_iter=200, T=1.0, tol=0.10,
    label="Test 5: Longer time T=1.0")

println("\n" * "="^70)
println("Summary:")
println("  Test 1 (α=0, con vs Enzyme):      ", pass1 ? "PASS" : "FAIL")
println("  Test 2 (α>0, con vs Enzyme):       ", pass2 ? "PASS" : "FAIL")
println("  Test 3 (IC sweep, α=0):            ", pass3 ? "PASS" : "FAIL")
println("  Test 4 (α sweep):                  ", pass4 ? "PASS" : "FAIL")
println("  Test 5 (T=1.0, long time):         ", pass5 ? "PASS" : "FAIL")
all_pass = pass1 && pass2 && pass3 && pass4 && pass5
println(all_pass ? "\nAll tests passed!" : "\nSome tests FAILED!")