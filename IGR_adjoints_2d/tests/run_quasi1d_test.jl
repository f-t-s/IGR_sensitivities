#!/usr/bin/env julia
# ============================================================
# Quasi-1D cross-validation: a 2D run with y-uniform fields and
# μy = 0 must reproduce the 1D package (IGRAdjoints1D) exactly.
#
# Rationale: the y-derivative of a y-uniform field is exactly
# zero (the LGL differentiation matrix has exact zero row sums),
# y-faces see zero jumps, and the 2D SIP operator and its RHS
# both scale by the same y-quadrature weight w[j]·Jy — leaving
# the elliptic solution unchanged. Hence the 2D solver restricted
# to quasi-1D data is the 1D solver, up to floating-point roundoff.
#
# This is a far tighter check than the FD/Enzyme comparisons:
# agreement is expected at ~1e-10, not at the OtD/FD tolerance.
# ============================================================

using Printf

include(joinpath(@__DIR__, "..", "..", "IGR_adjoints_1d", "IGRAdjoints1D.jl"))
include(joinpath(@__DIR__, "..", "IGRAdjoints2D.jl"))
const A1 = IGRAdjoints1D
const A2 = IGRAdjoints2D

# ------------------------------------------------------------
# Shared problem parameters
# ------------------------------------------------------------
γ   = 1.4
L   = 1.0
p   = 3
N_e = 12
A   = 0.4
T   = 0.08
CFL = 0.5
# Use PCG: it converges to the exact elliptic solution regardless of the
# preconditioner. The :chebyshev / :jacobi fixed-iteration solvers use
# 1D-tuned parameters and under-converge in 2D, which would mask the
# (exact) quasi-1D ≡ 1D equivalence behind an incomplete-solve gap.
solver = :pcg

N_ey = 4          # number of y-elements in the 2D run (arbitrary)
Ly   = 0.5        # y-domain length (arbitrary; fields are y-uniform)

# 1D and 2D discretizations sharing the x-direction exactly
b1 = A1.DGBasis(p);  m1 = A1.PeriodicMesh1D(N_e, L, b1)
b2 = A2.DGBasis(p);  m2 = A2.CartesianMesh2D(N_e, N_ey, L, Ly, b2)
n_p = p + 1

# ------------------------------------------------------------
# Velocity-pulse IC: u = A sin(2πx/L), ρ = 1, p = 1
# ------------------------------------------------------------
μ1  = zeros(n_p, N_e); ρ1 = zeros(n_p, N_e); E1 = zeros(n_p, N_e)
A1.build_velocity_pulse_ic!(μ1, ρ1, E1, A, m1.x, L, γ)

μx2 = similar(m2.x); μy2 = zeros(size(m2.x))
ρ2  = similar(m2.x); E2  = similar(m2.x)
for idx in eachindex(m2.x)
    s = sin(2π * m2.x[idx] / L)
    u = A * s
    ρ2[idx]  = 1.0
    μx2[idx] = u
    E2[idx]  = 1.0 / (γ - 1) + 0.5 * u^2
end

# Time step (identical for both runs)
max_ws = 0.0
for e in 1:N_e, i in 1:n_p
    global max_ws
    max_ws = max(max_ws, A1.max_wavespeed(γ, μ1[i,e], ρ1[i,e], E1[i,e]))
end
Δt = CFL * m1.Δx / ((2*p + 1) * max_ws)
n_steps = ceil(Int, T / Δt)
Δt = T / n_steps
α  = 3.0 * (L / N_e)^2
n_iter = 400   # enough PCG iterations for a machine-precision elliptic solve

@printf("Quasi-1D cross-validation:  p=%d, N_e=%d, N_ey=%d, n_steps=%d, α=%.4e\n",
        p, N_e, N_ey, n_steps, α)

# ------------------------------------------------------------
# Compare a 2D quasi-1D field against its 1D counterpart.
# Returns (max deviation from 1D, max y-nonuniformity).
# ------------------------------------------------------------
function compare(f2, f1)
    dev = 0.0; ynu = 0.0
    for ey in 1:N_ey, ex in 1:N_e, j in 1:n_p, i in 1:n_p
        dev = max(dev, abs(f2[i,j,ex,ey] - f1[i,ex]))
        ynu = max(ynu, abs(f2[i,j,ex,ey] - f2[i,1,ex,1]))
    end
    return dev, ynu
end

amax(f) = maximum(abs, f)

passed = true
function check(label, f2, f1)
    global passed
    dev, ynu = compare(f2, f1)
    ok = dev < 1e-9 && ynu < 1e-11
    passed &= ok
    @printf("    %-6s |2D-1D|=%.2e  y-nonunif=%.2e  %s\n",
            label, dev, ynu, ok ? "OK" : "FAIL")
end

# ============================================================
# Test 1: forward solve, pure Euler (α = 0)
# ============================================================
println("\nTest 1: forward solve, α = 0")
μf1, ρf1, Ef1, _ = A1.run_forward(μ1, ρ1, E1, Δt, T, b1, m1, γ, 0.0; n_iter=1, solver=solver)
μxf2, μyf2, ρf2, Ef2, _ = A2.run_forward(μx2, μy2, ρ2, E2, Δt, T, b2, m2, γ, 0.0; n_iter=1, solver=solver)
check("μx", μxf2, μf1)
check("ρ",  ρf2,  ρf1)
check("E",  Ef2,  Ef1)
@printf("    μy:    max|μy|=%.2e  %s\n", amax(μyf2), amax(μyf2) < 1e-11 ? "OK" : "FAIL")
passed &= amax(μyf2) < 1e-11

# ============================================================
# Test 2: forward solve with IGR (α > 0)
# ============================================================
println("\nTest 2: forward solve, α > 0 (IGR elliptic coupling)")
μf1, ρf1, Ef1, Σf1 = A1.run_forward(μ1, ρ1, E1, Δt, T, b1, m1, γ, α; n_iter=n_iter, solver=solver)
μxf2, μyf2, ρf2, Ef2, Σf2 = A2.run_forward(μx2, μy2, ρ2, E2, Δt, T, b2, m2, γ, α; n_iter=n_iter, solver=solver)
check("μx", μxf2, μf1)
check("ρ",  ρf2,  ρf1)
check("E",  Ef2,  Ef1)
check("Σ",  Σf2,  Σf1)
@printf("    μy:    max|μy|=%.2e  %s\n", amax(μyf2), amax(μyf2) < 1e-11 ? "OK" : "FAIL")
passed &= amax(μyf2) < 1e-11

# ============================================================
# Test 3: continuous adjoint PDE (α > 0)
# ============================================================
println("\nTest 3: adjoint PDE, α > 0  (objective = l2_density)")
snap1, ns1 = A1.run_forward_store(μ1, ρ1, E1, Δt, T, b1, m1, γ, α; n_iter=n_iter, solver=solver)
aμ1, aρ1, aE1 = A1.run_adjoint_conservative(snap1, ns1, Δt, T, b1, m1, γ, α;
    n_iter=n_iter, objective=:l2_density, solver=solver)
snap2, ns2 = A2.run_forward_store(μx2, μy2, ρ2, E2, Δt, T, b2, m2, γ, α; n_iter=n_iter, solver=solver)
aμx2, aμy2, aρ2, aE2 = A2.run_adjoint_conservative(snap2, ns2, Δt, T, b2, m2, γ, α;
    n_iter=n_iter, objective=:l2_density, solver=solver)
check("a_μx", aμx2, aμ1)
check("a_ρ",  aρ2,  aρ1)
check("a_E",  aE2,  aE1)
@printf("    a_μy:  max|a_μy|=%.2e  %s\n", amax(aμy2), amax(aμy2) < 1e-11 ? "OK" : "FAIL")
passed &= amax(aμy2) < 1e-11

# ============================================================
# Test 4: forward sensitivities (α > 0), θ = IC amplitude A
# ============================================================
println("\nTest 4: forward sensitivities, α > 0  (θ = IC amplitude)")
# 1D sensitivity IC:  ∂_A μ = sin(2πx/L), ∂_A ρ = 0, ∂_A E = A sin²
sμ1 = zeros(n_p, N_e); sρ1 = zeros(n_p, N_e); sE1 = zeros(n_p, N_e)
for e in 1:N_e, i in 1:n_p
    s = sin(2π * m1.x[i,e] / L)
    sμ1[i,e] = s
    sE1[i,e] = A * s^2
end
ζx20 = similar(m2.x); ζy20 = zeros(size(m2.x))
σ20  = zeros(size(m2.x)); η20 = similar(m2.x)
for idx in eachindex(m2.x)
    s = sin(2π * m2.x[idx] / L)
    ζx20[idx] = s
    η20[idx]  = A * s^2
end

_, _, _, _, sμf1, sρf1, sEf1, _ = A1.run_sensitivity(μ1, ρ1, E1, sμ1, sρ1, sE1,
    Δt, T, b1, m1, γ, α; n_iter=n_iter, elliptic_rhs=:form2, solver=solver, sγ=0.0)
_, _, _, _, _, ζxf2, ζyf2, σf2, ηf2, _ = A2.run_sensitivity(μx2, μy2, ρ2, E2,
    ζx20, ζy20, σ20, η20, Δt, T, b2, m2, γ, α; n_iter=n_iter, solver=solver)
check("ζx", ζxf2, sμf1)
check("σ",  σf2,  sρf1)
check("η",  ηf2,  sEf1)
@printf("    ζy:    max|ζy|=%.2e  %s\n", amax(ζyf2), amax(ζyf2) < 1e-11 ? "OK" : "FAIL")
passed &= amax(ζyf2) < 1e-11

println()
if passed
    println("All quasi-1D cross-validation tests passed (2D ≡ 1D to ~1e-10).")
else
    error("Quasi-1D cross-validation FAILED — 2D does not reduce to 1D.")
end
