#!/usr/bin/env julia
# ============================================================
# Test driver for the 2D doubly-periodic tensor-product DG
# IGR Euler solver.
# ============================================================

using Printf

include("../IGRAdjoints2D.jl")
using .IGRAdjoints2D

# ============================================================
# Test 1: Free-stream preservation
# ============================================================
function test_freestream()
    println("="^60)
    println("Test 1: Free-stream preservation")
    println("="^60)
    γ = 1.4
    for p in 1:4
        basis = DGBasis(p)
        mesh  = CartesianMesh2D(6, 5, 1.0, 1.3, basis)
        μx, μy, ρ, E = init_constant_state(mesh, γ; ux0=1.0, uy0=-0.7)
        Σ = zeros(size(μx))
        dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)
        compute_hyperbolic_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ)
        err = max(maximum(abs.(dμx)), maximum(abs.(dμy)),
                  maximum(abs.(dρ)),  maximum(abs.(dE)))
        @printf("  p=%d: max|dq| = %.2e\n", p, err)
        @assert err < 1e-11 "Free-stream not preserved for p=$p"
    end
    println("  PASSED\n")
end

# ============================================================
# Test 2: SIP elliptic solver (manufactured solution)
# ============================================================
function test_elliptic_manufactured()
    println("="^60)
    println("Test 2: 2D SIP elliptic solver (manufactured solution)")
    println("="^60)
    # Σ_exact = sin(2πx/Lx) sin(2πy/Ly), ρ = 1
    # Σ/ρ - α ∇·∇Σ = (1 + α k²) Σ,  k² = (2π/Lx)² + (2π/Ly)²
    α  = 0.1
    Lx = 1.0; Ly = 1.0
    for p in 1:4
        println("  p = $p:")
        errors = Float64[]
        N_es = [4, 8, 16, 32]
        for N_e in N_es
            basis = DGBasis(p)
            mesh  = CartesianMesh2D(N_e, N_e, Lx, Ly, basis)
            ρ = ones(size(mesh.x))
            Σ = zeros(size(mesh.x))
            Σ_exact = similar(Σ); b = similar(Σ)
            k2 = (2π/Lx)^2 + (2π/Ly)^2
            for idx in eachindex(mesh.x)
                Σ_exact[idx] = sin(2π*mesh.x[idx]/Lx) * sin(2π*mesh.y[idx]/Ly)
            end
            n_p = p + 1
            for ey in 1:N_e, ex in 1:N_e, j in 1:n_p, i in 1:n_p
                b[i,j,ex,ey] = basis.w[i]*basis.w[j]*mesh.Jx*mesh.Jy *
                               (1 + α*k2) * Σ_exact[i,j,ex,ey]
            end
            apply_A!(y, x) = apply_sip!(y, x, ρ, basis, mesh, α)
            iters = cg_solve!(Σ, apply_A!, b, 1e-12, 5000)
            err = l2_error(Σ, Σ_exact, basis, mesh)
            push!(errors, err)
            @printf("    N_e=%3d: L2 error = %.4e, CG iters = %d\n", N_e, err, iters)
        end
        for i in 2:length(errors)
            if errors[i] > 1e-13
                order = log(errors[i-1]/errors[i]) / log(2)
                @printf("    Order (N_e %d→%d): %.2f\n", N_es[i-1], N_es[i], order)
            end
        end
        println()
    end
end

# ============================================================
# Test 3: Isentropic vortex convergence (order p+1)
# ============================================================
function test_vortex_convergence()
    println("="^60)
    println("Test 3: Isentropic vortex convergence (α = 0)")
    println("="^60)
    γ = 1.4
    L = 10.0
    T = L  # one full diagonal period with ux0 = uy0 = 1
    for p in 1:3
        println("  p = $p:")
        errors = Float64[]
        N_es = [8, 16, 32, 64]
        for N_e in N_es
            basis = DGBasis(p)
            mesh  = CartesianMesh2D(N_e, N_e, L, L, basis)
            μx0, μy0, ρ0, E0 = init_isentropic_vortex(mesh, γ; β=5.0, ux0=1.0, uy0=1.0)

            max_ws = 0.0
            for idx in eachindex(μx0)
                ws = max(max_wavespeed_x(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]),
                         max_wavespeed_y(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]))
                max_ws = max(max_ws, ws)
            end
            Δt = 0.4 * min(mesh.Δx, mesh.Δy) / ((2*p + 1) * max_ws)

            μxf, μyf, ρf, Ef, _ = run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, 0.0; n_iter=0)
            err = l2_error(ρf, ρ0, basis, mesh)
            push!(errors, err)
            @printf("    N_e=%2d: L2(ρ) = %.4e\n", N_e, err)
        end
        for i in 2:length(errors)
            order = log(errors[i-1]/errors[i]) / log(2)
            @printf("    Order (N_e %d→%d): %.2f\n", N_es[i-1], N_es[i], order)
        end
        println()
    end
end

# ============================================================
# Test 4: IGR Euler — Taylor-Green with α > 0
# ============================================================
function test_igr_taylor_green()
    println("="^60)
    println("Test 4: IGR Euler — Taylor-Green vortex with α > 0")
    println("="^60)
    γ = 1.4
    L = 1.0
    T = 0.3
    p = 3
    N_e = 16
    A = 0.5

    basis = DGBasis(p)
    mesh  = CartesianMesh2D(N_e, N_e, L, L, basis)
    μx0 = similar(mesh.x); μy0 = similar(mesh.x)
    ρ0  = similar(mesh.x); E0  = similar(mesh.x)
    build_taylor_green_ic!(μx0, μy0, ρ0, E0, A, mesh, γ)

    max_ws = 0.0
    for idx in eachindex(μx0)
        ws = max(max_wavespeed_x(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]),
                 max_wavespeed_y(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]))
        max_ws = max(max_ws, ws)
    end
    Δt = 0.3 * min(mesh.Δx, mesh.Δy) / ((2*p + 1) * max_ws)

    α = 3.0 * (L / N_e)^2
    for solver in (:pcg, :chebyshev, :jacobi)
        μxf, μyf, ρf, Ef, Σf = run_forward(μx0, μy0, ρ0, E0, Δt, T, basis, mesh, γ, α;
                                            n_iter=80, solver=solver)
        @printf("  solver=%-10s ρ∈[%.4f,%.4f]  max|Σ|=%.4e\n",
                solver, minimum(ρf), maximum(ρf), maximum(abs.(Σf)))
        @assert all(isfinite, ρf) "Solution blew up (solver=$solver)"
        @assert minimum(ρf) > 0   "Negative density (solver=$solver)"
        @assert maximum(abs.(Σf)) > 0 "Σ should be nonzero (solver=$solver)"
    end
    println("  PASSED\n")
end

test_freestream()
test_elliptic_manufactured()
test_vortex_convergence()
test_igr_taylor_green()
println("All Stage 1 tests completed.")
