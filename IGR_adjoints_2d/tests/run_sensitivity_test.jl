#!/usr/bin/env julia
# ============================================================
# Verify the 2D forward sensitivity solver against central
# finite differences. The parameter θ is the amplitude A of
# the Taylor-Green initial condition.
# ============================================================

using Printf, LinearAlgebra

include("../IGRAdjoints2D.jl")
using .IGRAdjoints2D

# ∂_A of the Taylor-Green initial condition (ρ₀ = 1 ⇒ σ₀ = 0)
function taylor_green_sens_ic(mesh, γ, A)
    ζx0 = similar(mesh.x); ζy0 = similar(mesh.x)
    σ0  = zeros(size(mesh.x)); η0 = similar(mesh.x)
    for idx in eachindex(mesh.x)
        cx = 2π * mesh.x[idx] / mesh.Lx
        cy = 2π * mesh.y[idx] / mesh.Ly
        ζx0[idx] = sin(cx) * cos(cy)            # ∂_A μx₀
        ζy0[idx] = -cos(cx) * sin(cy)           # ∂_A μy₀
        η0[idx]  = A * (sin(cx)^2 * cos(cy)^2 + cos(cx)^2 * sin(cy)^2)  # ∂_A E₀
    end
    return ζx0, ζy0, σ0, η0
end

function run_test(α::Float64; p=3, N_e=8, T=0.1)
    γ = 1.4
    L = 1.0
    A = 0.4
    solver = :chebyshev

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
    Δt = 0.4 * min(mesh.Δx, mesh.Δy) / ((2*p + 1) * max_ws)
    n_iter = α > 0 ? 100 : 1

    # --- Forward sensitivity solve ---
    ζx0, ζy0, σ0, η0 = taylor_green_sens_ic(mesh, γ, A)
    _, _, _, _, _, ζx, ζy, σ, η, _ = run_sensitivity(μx0, μy0, ρ0, E0,
        ζx0, ζy0, σ0, η0, Δt, T, basis, mesh, γ, α; n_iter=n_iter, solver=solver)

    # --- Central finite differences in A ---
    h = 1e-5
    function fwd(Aval)
        mx = similar(mesh.x); my = similar(mesh.x)
        r  = similar(mesh.x); e  = similar(mesh.x)
        build_taylor_green_ic!(mx, my, r, e, Aval, mesh, γ)
        return run_forward(mx, my, r, e, Δt, T, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
    end
    μxp, μyp, ρp, Ep, _ = fwd(A + h)
    μxm, μym, ρm, Em, _ = fwd(A - h)
    fd_ζx = (μxp .- μxm) ./ (2h)
    fd_ζy = (μyp .- μym) ./ (2h)
    fd_σ  = (ρp  .- ρm)  ./ (2h)
    fd_η  = (Ep  .- Em)  ./ (2h)

    relerr(s, f) = norm(s .- f) / max(norm(f), 1e-30)
    eζx = relerr(ζx, fd_ζx); eζy = relerr(ζy, fd_ζy)
    eσ  = relerr(σ,  fd_σ);  eη  = relerr(η,  fd_η)
    combined = norm(vcat(vec(ζx .- fd_ζx), vec(ζy .- fd_ζy),
                         vec(σ .- fd_σ),   vec(η .- fd_η))) /
               max(norm(vcat(vec(fd_ζx), vec(fd_ζy), vec(fd_σ), vec(fd_η))), 1e-30)

    @printf("  α=%.1e:  rel L²  ζx=%.3e ζy=%.3e σ=%.3e η=%.3e   combined=%.3e\n",
            α, eζx, eζy, eσ, eη, combined)
    # Tolerance reflects central-difference accuracy plus the elliptic
    # solver's finite iteration count; a bug yields O(1) disagreement.
    @assert combined < 5e-4 "Sensitivity disagrees with FD (α=$α): $combined"
    println("    PASSED")
    return nothing
end

println("="^60)
println("2D forward sensitivity  vs  finite differences (θ = IC amplitude)")
println("="^60)
run_test(0.0)
run_test(1.5e-3)
println("\nAll Stage 4 sensitivity tests passed.")
