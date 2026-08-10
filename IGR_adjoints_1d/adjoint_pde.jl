# ============================================================
# Continuous adjoint PDE solver for the IGR Euler system
#
# Implements the "optimize-then-discretize" adjoint approach
# from adjoints.tex using the conservative PDE formulation
# with the adjoint elliptic variable Π.
#
# The adjoint equation splits into local (P-dependent) and
# nonlocal (Σ-dependent) parts:
#   ∂_τ q† = div(F†_loc) - S_loc + S_Σ
# where F†_loc uses only physical P derivatives (not P̄),
# S_loc is the local source with P derivatives only,
# and S_Σ is the nonlocal source computed from Π via
# a single adjoint elliptic solve.
# ============================================================

export adjoint_flux_local, adjoint_llf_flux_local,
       compute_adjoint_source_local!,
       compute_adjoint_hyperbolic_rhs_local!,
       compute_adjoint_elliptic_rhs!,
       compute_nonlocal_source!,
       compute_adjoint_rhs_conservative!,
       run_adjoint_conservative

# ============================================================
# Pressure derivatives (polytropic EOS, 1D)
# ============================================================

# ∂_E P = γ - 1
# ∂_μ P = -(γ-1)u
# ∂_ρ P = (γ-1)|u|²/2

# ============================================================
# Local adjoint flux (P derivatives only, no Σ_ρ, Σ_μ)
# ============================================================

"""
    adjoint_flux_local(γ, μ, ρ, E, Σ, aμ, aρ, aE)

Local adjoint flux using only physical pressure P derivatives.
The entropic pressure Σ appears in the flux through P̄ = P + Σ,
but its derivatives w.r.t. q are NOT included (Σ is frozen).

This is F†_loc from eq. (868) in adjoints.tex.
"""
function adjoint_flux_local(γ, μ, ρ, E, Σ, aμ, aρ, aE)
    u = μ / ρ
    P = polytropic_pressure(γ, μ, ρ, E)
    P̄ = P + Σ

    # Physical pressure derivatives only (eq. 884-888)
    dP_dE = γ - 1
    dP_dμ = -(γ - 1) * u
    dP_dρ = (γ - 1) * u^2 / 2

    # F†_μ = aμ·(2u + ∂_μ P) + aρ + aE·((E+P̄)/ρ + ∂_μ P·u)
    f_aμ = aμ * (2 * u + dP_dμ) + aρ + aE * ((E + P̄) / ρ + dP_dμ * u)

    # F†_ρ = aμ·(-u² + ∂_ρ P) + aE·(∂_ρ P · u - (E+P̄)·u/ρ)
    f_aρ = aμ * (-u^2 + dP_dρ) + aE * (dP_dρ * u - (E + P̄) * u / ρ)

    # F†_E = aμ·∂_E P + aE·(1 + ∂_E P)·u
    f_aE = aμ * dP_dE + aE * (1 + dP_dE) * u

    return (f_aμ, f_aρ, f_aE)
end

"""
    adjoint_llf_flux_local(γ, μL, ρL, EL, ΣL,
                           μR, ρR, ER, ΣR,
                           aμL, aρL, aEL, aμR, aρR, aER)

LLF numerical flux for the local adjoint system (no Σ_ρ, Σ_μ).
"""
function adjoint_llf_flux_local(γ, μL, ρL, EL, ΣL,
                                μR, ρR, ER, ΣR,
                                aμL, aρL, aEL, aμR, aρR, aER)
    λ = max(max_wavespeed(γ, μL, ρL, EL), max_wavespeed(γ, μR, ρR, ER))

    fμL, fρL, fEL = adjoint_flux_local(γ, μL, ρL, EL, ΣL, aμL, aρL, aEL)
    fμR, fρR, fER = adjoint_flux_local(γ, μR, ρR, ER, ΣR, aμR, aρR, aER)

    f_aμ = (fμL + fμR) / 2 - λ * (aμL - aμR) / 2
    f_aρ = (fρL + fρR) / 2 - λ * (aρL - aρR) / 2
    f_aE = (fEL + fER) / 2 - λ * (aEL - aER) / 2

    return (f_aμ, f_aρ, f_aE)
end

# ============================================================
# Local adjoint source (P derivatives only)
# ============================================================

"""
    compute_adjoint_source_local!(Sμ, Sρ, SE, aμ, aρ, aE,
        μ, ρ, E, Σ, basis, mesh, γ)

Compute the local adjoint source using only physical P derivatives.
Same structure as `compute_adjoint_source!` but with Σ_ρ=Σ_μ=0
and their spatial derivatives set to zero.
"""
function compute_adjoint_source_local!(Sμ, Sρ, SE, aμ, aρ, aE,
        μ, ρ, E, Σ, basis, mesh, γ)
    N_e = mesh.N_e
    n_p = basis.p + 1
    D = basis.D
    invJ = 1 / mesh.J

    for e in 1:N_e
        for i in 1:n_p
            # Spatial derivatives of primal variables
            μ_x = zero(eltype(μ))
            ρ_x = zero(eltype(ρ))
            E_x = zero(eltype(E))
            Σ_x = zero(eltype(Σ))

            for j in 1:n_p
                μ_x  += D[i,j] * μ[j,e]
                ρ_x  += D[i,j] * ρ[j,e]
                E_x  += D[i,j] * E[j,e]
                Σ_x  += D[i,j] * Σ[j,e]
            end
            μ_x  *= invJ
            ρ_x  *= invJ
            E_x  *= invJ
            Σ_x  *= invJ

            ρ_i = ρ[i,e]
            μ_i = μ[i,e]
            E_i = E[i,e]
            u_i = μ_i / ρ_i
            P_i = polytropic_pressure(γ, μ_i, ρ_i, E_i)
            P̄_i = P_i + Σ[i,e]

            u_x = (μ_x - u_i * ρ_x) / ρ_i

            # Physical pressure derivatives only (Σ dependence on q handled by Π)
            dP_dE = γ - 1
            dP_dμ = -(γ - 1) * u_i
            dP_dρ = (γ - 1) * u_i^2 / 2

            # Spatial derivatives of P (no Σ_ρ_x, Σ_μ_x — those are nonlocal)
            P_x = (γ - 1) * (E_x - μ_i * μ_x / ρ_i + μ_i^2 * ρ_x / (2 * ρ_i^2))
            dρP_x = (γ - 1) * (μ_i * μ_x / ρ_i^2 - μ_i^2 * ρ_x / ρ_i^3)
            dμP_x = -(γ - 1) * u_x

            # P̄_x includes Σ_x (Σ is a known frozen field)
            P̄_x = P_x + Σ_x

            aμ_i = aμ[i,e]
            aE_i = aE[i,e]

            # Parametric derivatives of u and u_x
            dμ_u  = 1 / ρ_i
            dμ_ux = -ρ_x / ρ_i^2
            dρ_u  = -u_i / ρ_i
            dρ_ux = -μ_x / ρ_i^2 + 2 * μ_i * ρ_x / ρ_i^3

            # ∂_μ(div_F) terms
            dμ_divFμ = μ_x * dμ_u + u_x + μ_i * dμ_ux + dμP_x
            dμ_divFE = dμP_x * u_i + (E_x + P̄_x) * dμ_u + dP_dμ * u_x + (E_i + P̄_i) * dμ_ux
            Sμ[i,e] = aμ_i * dμ_divFμ + aE_i * dμ_divFE

            # ∂_ρ(div_F) terms
            dρ_divFμ = μ_x * dρ_u + μ_i * dρ_ux + dρP_x
            dρ_divFE = dρP_x * u_i + (E_x + P̄_x) * dρ_u + dP_dρ * u_x + (E_i + P̄_i) * dρ_ux
            Sρ[i,e] = aμ_i * dρ_divFμ + aE_i * dρ_divFE

            # ∂_E(div_F) terms
            dE_divFE = (1 + dP_dE) * u_x
            SE[i,e] = aE_i * dE_divFE
        end
    end

    return nothing
end

# ============================================================
# Local adjoint DG operator (flux + source, no Σ coupling)
# ============================================================

"""
    compute_adjoint_hyperbolic_rhs_local!(daμ, daρ, daE,
        aμ, aρ, aE, μ, ρ, E, Σ, basis, mesh, γ)

DG right-hand side for the local adjoint equation:
  ∂_τ q† = +div(F†_loc) - S_loc

Uses only physical P derivatives (no Σ_ρ, Σ_μ).
"""
function compute_adjoint_hyperbolic_rhs_local!(daμ, daρ, daE,
        aμ, aρ, aE, μ, ρ, E, Σ, basis, mesh, γ)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w
    invJ = 1 / J

    # Compute local adjoint flux at all nodes
    fμ = similar(aμ)
    fρ = similar(aρ)
    fE = similar(aE)
    for e in 1:N_e, i in 1:n_p
        fμ[i,e], fρ[i,e], fE[i,e] = adjoint_flux_local(γ,
            μ[i,e], ρ[i,e], E[i,e], Σ[i,e],
            aμ[i,e], aρ[i,e], aE[i,e])
    end

    # Volume terms: dq† = +(1/J) D f†_loc
    for e in 1:N_e, i in 1:n_p
        daμ[i,e] = zero(eltype(aμ))
        daρ[i,e] = zero(eltype(aρ))
        daE[i,e] = zero(eltype(aE))
        for j in 1:n_p
            daμ[i,e] += invJ * D[i,j] * fμ[j,e]
            daρ[i,e] += invJ * D[i,j] * fρ[j,e]
            daE[i,e] += invJ * D[i,j] * fE[j,e]
        end
    end

    # Face fluxes and corrections
    for e in 1:N_e
        e_left  = e == 1 ? N_e : e - 1
        e_right = e == N_e ? 1 : e + 1

        # Right face
        fstar_aμR, fstar_aρR, fstar_aER = adjoint_llf_flux_local(γ,
            μ[n_p,e], ρ[n_p,e], E[n_p,e], Σ[n_p,e],
            μ[1,e_right], ρ[1,e_right], E[1,e_right], Σ[1,e_right],
            aμ[n_p,e], aρ[n_p,e], aE[n_p,e],
            aμ[1,e_right], aρ[1,e_right], aE[1,e_right])

        # Left face
        fstar_aμL, fstar_aρL, fstar_aEL = adjoint_llf_flux_local(γ,
            μ[n_p,e_left], ρ[n_p,e_left], E[n_p,e_left], Σ[n_p,e_left],
            μ[1,e], ρ[1,e], E[1,e], Σ[1,e],
            aμ[n_p,e_left], aρ[n_p,e_left], aE[n_p,e_left],
            aμ[1,e], aρ[1,e], aE[1,e])

        # Right face correction (positive sign)
        daμ[n_p,e] += invJ * (fstar_aμR - fμ[n_p,e]) / w[n_p]
        daρ[n_p,e] += invJ * (fstar_aρR - fρ[n_p,e]) / w[n_p]
        daE[n_p,e] += invJ * (fstar_aER - fE[n_p,e]) / w[n_p]

        # Left face correction (negative sign)
        daμ[1,e] -= invJ * (fstar_aμL - fμ[1,e]) / w[1]
        daρ[1,e] -= invJ * (fstar_aρL - fρ[1,e]) / w[1]
        daE[1,e] -= invJ * (fstar_aEL - fE[1,e]) / w[1]
    end

    # Subtract local source term
    Sμ = similar(aμ)
    Sρ = similar(aρ)
    SE = similar(aE)
    compute_adjoint_source_local!(Sμ, Sρ, SE, aμ, aρ, aE,
        μ, ρ, E, Σ, basis, mesh, γ)

    @. daμ -= Sμ
    @. daρ -= Sρ
    @. daE -= SE

    return nothing
end

# ============================================================
# RHS for the adjoint elliptic solve (Π equation)
# ============================================================

"""
    compute_adjoint_elliptic_rhs!(b, aμ, aE, μ, ρ, basis, mesh)

Compute the RHS integrand for the adjoint elliptic equation (eq. 949 in adjoints.tex):
  a_h(Π, ψ) = α ∫ b ψ dx

where b = div(q†_μ) + [D q†_E]·μ/ρ.  In 1D: b = ∂_x(aμ) + ∂_x(aE)·u.
"""
function compute_adjoint_elliptic_rhs!(b, aμ, aE, μ, ρ, basis, mesh)
    N_e = mesh.N_e
    n_p = basis.p + 1
    D = basis.D
    invJ = 1 / mesh.J

    for e in 1:N_e
        for i in 1:n_p
            daμ_dx = zero(eltype(aμ))
            daE_dx = zero(eltype(aE))
            for j in 1:n_p
                daμ_dx += D[i,j] * aμ[j,e]
                daE_dx += D[i,j] * aE[j,e]
            end
            daμ_dx *= invJ
            daE_dx *= invJ

            u_i = μ[i,e] / ρ[i,e]
            b[i,e] = daμ_dx + daE_dx * u_i
        end
    end

    return nothing
end

# ============================================================
# Nonlocal source S_Σ from the adjoint elliptic variable Π
# ============================================================

"""
    compute_nonlocal_source!(Sμ, Sρ, SE, Π, μ, ρ, Σ, basis, mesh)

Compute the nonlocal source term S_Σ from the adjoint elliptic
variable Π (eq. 926 in adjoints.tex).

In 1D, the formulas simplify to:
  S_Σ,μ = -4·∂_x(Π·u_x)/ρ
  S_Σ,ρ = [2Π·u_x² + ∂_x(Π·Σ_x/ρ)]/ρ + 4·∂_x(Π·u_x)·μ/ρ²
  S_Σ,E = 0

This requires "double differentiation": first compute u_x, Σ_x,
then form products Π·u_x and Π·Σ_x/ρ, then differentiate again.
"""
function compute_nonlocal_source!(Sμ, Sρ, SE, Π, μ, ρ, Σ, basis, mesh)
    N_e = mesh.N_e
    n_p = basis.p + 1
    D = basis.D
    invJ = 1 / mesh.J

    # First pass: compute u_x and Σ_x/ρ at all nodes, then form products
    Pi_ux = similar(Π)    # Π · u_x
    Pi_Sx_r = similar(Π)  # Π · Σ_x / ρ

    for e in 1:N_e
        for i in 1:n_p
            # Compute u_x and Σ_x at node i
            u_x_i = zero(eltype(μ))
            Σ_x_i = zero(eltype(Σ))
            for j in 1:n_p
                u_j = μ[j,e] / ρ[j,e]
                u_x_i += D[i,j] * u_j
                Σ_x_i += D[i,j] * Σ[j,e]
            end
            u_x_i *= invJ
            Σ_x_i *= invJ

            Pi_ux[i,e] = Π[i,e] * u_x_i
            Pi_Sx_r[i,e] = Π[i,e] * Σ_x_i / ρ[i,e]
        end
    end

    # Second pass: differentiate the products and assemble source
    for e in 1:N_e
        for i in 1:n_p
            # ∂_x(Π·u_x) at node i
            d_Pi_ux = zero(eltype(Π))
            # ∂_x(Π·Σ_x/ρ) at node i
            d_Pi_Sx_r = zero(eltype(Π))

            for j in 1:n_p
                d_Pi_ux   += D[i,j] * Pi_ux[j,e]
                d_Pi_Sx_r += D[i,j] * Pi_Sx_r[j,e]
            end
            d_Pi_ux   *= invJ
            d_Pi_Sx_r *= invJ

            # Recompute u_x for the algebraic term 2Π·u_x²
            u_x_i = zero(eltype(μ))
            for j in 1:n_p
                u_x_i += D[i,j] * (μ[j,e] / ρ[j,e])
            end
            u_x_i *= invJ

            ρ_i = ρ[i,e]
            μ_i = μ[i,e]

            # S_Σ,μ = -4·∂_x(Π·u_x)/ρ
            Sμ[i,e] = -4 * d_Pi_ux / ρ_i

            # S_Σ,ρ = [2Π·u_x² + ∂_x(Π·Σ_x/ρ)]/ρ + 4·∂_x(Π·u_x)·μ/ρ²
            Sρ[i,e] = (2 * Π[i,e] * u_x_i^2 + d_Pi_Sx_r) / ρ_i +
                       4 * d_Pi_ux * μ_i / ρ_i^2

            # S_Σ,E = 0
            SE[i,e] = zero(eltype(Π))
        end
    end

    return nothing
end

# ============================================================
# Full conservative adjoint RHS (local + Π nonlocal source)
# ============================================================

"""
    compute_adjoint_rhs_conservative!(daμ, daρ, daE,
        aμ, aρ, aE, μ, ρ, E, Σ, Π,
        basis, mesh, γ, α; n_iter=10)

Full adjoint spatial operator using the conservative PDE approach
with the adjoint elliptic variable Π.

1. Compute the local adjoint RHS (flux + source with P derivatives only)
2. If α > 0: compute RHS for Π equation, solve for Π, compute S_Σ, add to RHS

The Π workspace array is passed in to avoid allocation.
"""
function compute_adjoint_rhs_conservative!(daμ, daρ, daE,
        aμ, aρ, aE, μ, ρ, E, Σ, Π,
        basis, mesh, γ, α; n_iter=10, solver=:pcg)
    # Step 1: Local adjoint (P derivatives only)
    compute_adjoint_hyperbolic_rhs_local!(daμ, daρ, daE,
        aμ, aρ, aE, μ, ρ, E, Σ, basis, mesh, γ)

    if α > 0
        # Step 2: Compute RHS for adjoint elliptic equation (eq. 949)
        n_p = basis.p + 1
        N_e = mesh.N_e
        w = basis.w
        J = mesh.J

        b_Pi = similar(Π)
        compute_adjoint_elliptic_rhs!(b_Pi, aμ, aE, μ, ρ, basis, mesh)

        # Step 3: Form discrete RHS: b = α · w · J · (∂_x(aμ) + ∂_x(aE)·u)
        for e in 1:N_e, i in 1:n_p
            b_Pi[i,e] *= α * w[i] * J
        end

        # Solve L[Π] = b_Pi using same SIP operator (Π warm-started from previous solve)
        M_diag = compute_sip_diagonal(ρ, basis, mesh, α)
        if solver == :pcg
            pcg_fixed!(Π, b_Pi, M_diag, n_iter, ρ, basis, mesh, α)
        elseif solver == :jacobi
            jacobi_fixed!(Π, b_Pi, M_diag, n_iter, ρ, basis, mesh, α, 2.0/3.0)
        elseif solver == :chebyshev
            chebyshev_fixed!(Π, b_Pi, M_diag, n_iter, ρ, basis, mesh, α)
        end

        # Step 4: Compute nonlocal source and add to RHS
        Sμ = similar(daμ)
        Sρ = similar(daρ)
        SE = similar(daE)
        compute_nonlocal_source!(Sμ, Sρ, SE, Π, μ, ρ, Σ, basis, mesh)

        # S_Σ is a source on the RHS: ∂_τ q† = div(F†) - S_loc + S_Σ
        @. daμ += Sμ
        @. daρ += Sρ
        @. daE += SE
    end

    return nothing
end

# ============================================================
# Backward-in-time adjoint integration (conservative approach)
# ============================================================

"""
    run_adjoint_conservative(snapshots, n_steps, Δt, T, basis, mesh, γ, α;
                             n_iter=10, objective=:l2_density)

Run the adjoint PDE backward in time using the conservative Π approach.

This requires only 1 elliptic solve per RK stage (for Π) instead of
d+1 solves (for Σ_ρ, Σ_μ) in the IFT approach.
"""
function run_adjoint_conservative(snapshots, n_steps, Δt, T, basis, mesh, γ, α;
                                  n_iter=10, objective=:l2_density, solver=:pcg)
    n_p = basis.p + 1
    N_e = mesh.N_e

    # Terminal condition
    _, ρT, _, _ = snapshots[end]
    RT = eltype(ρT)

    aμ = zeros(RT, n_p, N_e)
    aρ = zeros(RT, n_p, N_e)
    aE = zeros(RT, n_p, N_e)

    if objective == :l2_density
        @. aρ = 2 * ρT
    elseif objective == :l2_pressure
        μT, _, ET, _ = snapshots[end]
        uT = μT ./ ρT
        pT = (γ - 1) .* (ET .- μT.^2 ./ (2 .* ρT))
        @. aμ = -2 * (γ - 1) * pT * uT
        @. aρ = (γ - 1) * pT * uT^2
        @. aE = 2 * (γ - 1) * pT
    elseif objective == :weighted_momentum
        # J = ∫ μ(x,T) sin(2πx/L) dx
        # ∂J/∂μ = sin(2πx/L), ∂J/∂ρ = 0, ∂J/∂E = 0
        L_domain = N_e * mesh.Δx
        for e in 1:N_e, i in 1:n_p
            aμ[i,e] = sin(2π * mesh.x[i,e] / L_domain)
        end
    elseif objective == :kinetic_energy
        # J = KE/M = (∫ μ²/(2ρ) dx) / (∫ ρ dx)
        # ∂J/∂μ = (μ/ρ) / M
        # ∂J/∂ρ = -μ²/(2ρ²) / M - KE/M²
        # ∂J/∂E = 0
        μT, _, _, _ = snapshots[end]
        KE = zero(RT)
        M = zero(RT)
        for e in 1:N_e, i in 1:n_p
            wJ = basis.w[i] * mesh.J
            KE += wJ * μT[i,e]^2 / (2 * ρT[i,e])
            M  += wJ * ρT[i,e]
        end
        @. aμ = (μT / ρT) / M
        @. aρ = -μT^2 / (2 * ρT^2) / M - KE / M^2
        # aE stays zero
    else
        error("Unknown objective: $objective (use :l2_density, :l2_pressure, or :kinetic_energy)")
    end

    # Work array for adjoint elliptic variable
    Π = zeros(RT, n_p, N_e)

    # Stage arrays
    aμ1 = similar(aμ); aρ1 = similar(aρ); aE1 = similar(aE)
    aμ2 = similar(aμ); aρ2 = similar(aρ); aE2 = similar(aE)
    daμ = similar(aμ); daρ = similar(aρ); daE = similar(aE)

    # Integrate backward
    for n in n_steps:-1:1
        dt = min(Δt, T)
        μ_n, ρ_n, E_n, Σ_n = snapshots[n+1]

        function adj_rhs!(daμ, daρ, daE, aμ, aρ, aE)
            compute_adjoint_rhs_conservative!(daμ, daρ, daE,
                aμ, aρ, aE, μ_n, ρ_n, E_n, Σ_n, Π,
                basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        end

        # SSP-RK3
        adj_rhs!(daμ, daρ, daE, aμ, aρ, aE)
        @. aμ1 = aμ + dt * daμ
        @. aρ1 = aρ + dt * daρ
        @. aE1 = aE + dt * daE

        adj_rhs!(daμ, daρ, daE, aμ1, aρ1, aE1)
        @. aμ2 = 3/4 * aμ + 1/4 * aμ1 + 1/4 * dt * daμ
        @. aρ2 = 3/4 * aρ + 1/4 * aρ1 + 1/4 * dt * daρ
        @. aE2 = 3/4 * aE + 1/4 * aE1 + 1/4 * dt * daE

        adj_rhs!(daμ, daρ, daE, aμ2, aρ2, aE2)
        @. aμ = 1/3 * aμ + 2/3 * aμ2 + 2/3 * dt * daμ
        @. aρ = 1/3 * aρ + 2/3 * aρ2 + 2/3 * dt * daρ
        @. aE = 1/3 * aE + 2/3 * aE2 + 2/3 * dt * daE
    end

    return aμ, aρ, aE
end