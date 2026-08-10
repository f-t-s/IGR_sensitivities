# ============================================================
# Forward sensitivity equations for the IGR Euler system
#
# Discretizes the continuous sensitivity PDEs (adjoints.tex
# lines 167-180) using the same DG framework as the forward
# solver. Two forms of the elliptic sensitivity RHS are
# provided (:form1 and :form2), swappable via keyword argument.
# ============================================================

export sensitivity_flux, sensitivity_llf_flux,
       compute_sensitivity_hyperbolic_rhs!,
       compute_sensitivity_elliptic_rhs_form1!,
       compute_sensitivity_elliptic_rhs_form2!,
       solve_sensitivity_elliptic!,
       compute_sensitivity_rhs!,
       ssp_rk3_coupled,
       run_sensitivity

# ============================================================
# Pointwise sensitivity flux
# ============================================================

"""
    sensitivity_flux(γ, μ, ρ, E, Σ, sμ, sρ, sE, sΣ; sγ=0)

Compute the sensitivity flux at a single node. Returns (fsμ, fsρ, fsE).

The sensitivity conservation law (adjoints.tex lines 174-176):
  ∂_t ŝ_μ + ∂_x(ŝ_μ u + μ ŝ_u + ŝ_P + ŝ_Σ) = 0
  ∂_t ŝ_ρ + ∂_x(ŝ_μ) = 0
  ∂_t ŝ_E + ∂_x(ŝ_u(E+P+Σ) + u(ŝ_E + ŝ_P + ŝ_Σ)) = 0

The pressure sensitivity includes the direct ∂P/∂γ term (adjoints.tex line 149):
  ŝ_P = (γ-1)(ŝ_E - ŝ_ρ u²/2 - ρ u ŝ_u) + ŝ_γ ρ e
"""
function sensitivity_flux(γ, μ, ρ, E, Σ, sμ, sρ, sE, sΣ; sγ=zero(γ))
    u  = μ / ρ
    su = (sμ - u * sρ) / ρ
    P  = polytropic_pressure(γ, μ, ρ, E)
    e  = E / ρ - u^2 / 2
    sP = (γ - 1) * (sE - sρ * u^2 / 2 - ρ * u * su) + sγ * ρ * e

    fsμ = sμ * u + μ * su + sP + sΣ
    fsρ = sμ
    fsE = su * (E + P + Σ) + u * (sE + sP + sΣ)

    return (fsμ, fsρ, fsE)
end

# ============================================================
# Sensitivity LLF numerical flux
# ============================================================

"""
    sensitivity_llf_flux(γ, μL, ρL, EL, ΣL, μR, ρR, ER, ΣR,
                         sμL, sρL, sEL, sΣL, sμR, sρR, sER, sΣR; sγ=0)

LLF numerical flux for the sensitivity system. Uses the primal
wavespeed (the linearized system has the same characteristic speeds
as the primal Euler system).
"""
function sensitivity_llf_flux(γ, μL, ρL, EL, ΣL, μR, ρR, ER, ΣR,
                              sμL, sρL, sEL, sΣL, sμR, sρR, sER, sΣR; sγ=zero(γ))
    # Wavespeed from primal state
    λ = max(max_wavespeed(γ, μL, ρL, EL), max_wavespeed(γ, μR, ρR, ER))

    # Sensitivity fluxes on each side
    fsμL, fsρL, fsEL = sensitivity_flux(γ, μL, ρL, EL, ΣL, sμL, sρL, sEL, sΣL; sγ=sγ)
    fsμR, fsρR, fsER = sensitivity_flux(γ, μR, ρR, ER, ΣR, sμR, sρR, sER, sΣR; sγ=sγ)

    # LLF: f★ = (fL + fR)/2 + λ/2 (sL - sR)
    f_sμ = (fsμL + fsμR) / 2 + λ * (sμL - sμR) / 2
    f_sρ = (fsρL + fsρR) / 2 + λ * (sρL - sρR) / 2
    f_sE = (fsEL + fsER) / 2 + λ * (sEL - sER) / 2

    return (f_sμ, f_sρ, f_sE)
end

# ============================================================
# Sensitivity hyperbolic RHS (DG spatial operator)
# ============================================================

"""
    compute_sensitivity_hyperbolic_rhs!(dsμ, dsρ, dsE,
        μ, ρ, E, Σ, sμ, sρ, sE, sΣ, basis, mesh, γ; sγ=0)

DG right-hand side for the sensitivity hyperbolic equations.
Same strong-form DG structure as `compute_hyperbolic_rhs!`.
"""
function compute_sensitivity_hyperbolic_rhs!(dsμ, dsρ, dsE,
        μ, ρ, E, Σ, sμ, sρ, sE, sΣ, basis, mesh, γ; sγ=zero(γ))
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w

    # Compute sensitivity flux at all nodes
    fμ = similar(sμ)
    fρ = similar(sρ)
    fE = similar(sE)
    for e in 1:N_e, i in 1:n_p
        fμ[i,e], fρ[i,e], fE[i,e] = sensitivity_flux(γ,
            μ[i,e], ρ[i,e], E[i,e], Σ[i,e],
            sμ[i,e], sρ[i,e], sE[i,e], sΣ[i,e]; sγ=sγ)
    end

    # Volume terms: dq = -(1/J) D f
    invJ = 1 / J
    for e in 1:N_e, i in 1:n_p
        dsμ[i,e] = zero(eltype(sμ))
        dsρ[i,e] = zero(eltype(sρ))
        dsE[i,e] = zero(eltype(sE))
        for j in 1:n_p
            dsμ[i,e] -= invJ * D[i,j] * fμ[j,e]
            dsρ[i,e] -= invJ * D[i,j] * fρ[j,e]
            dsE[i,e] -= invJ * D[i,j] * fE[j,e]
        end
    end

    # Face fluxes and corrections
    for e in 1:N_e
        e_left  = e == 1 ? N_e : e - 1
        e_right = e == N_e ? 1 : e + 1

        # Right face: interface between e (right) and e_right (left)
        fstar_sμR, fstar_sρR, fstar_sER = sensitivity_llf_flux(γ,
            μ[n_p,e], ρ[n_p,e], E[n_p,e], Σ[n_p,e],
            μ[1,e_right], ρ[1,e_right], E[1,e_right], Σ[1,e_right],
            sμ[n_p,e], sρ[n_p,e], sE[n_p,e], sΣ[n_p,e],
            sμ[1,e_right], sρ[1,e_right], sE[1,e_right], sΣ[1,e_right]; sγ=sγ)

        # Left face: interface between e_left (right) and e (left)
        fstar_sμL, fstar_sρL, fstar_sEL = sensitivity_llf_flux(γ,
            μ[n_p,e_left], ρ[n_p,e_left], E[n_p,e_left], Σ[n_p,e_left],
            μ[1,e], ρ[1,e], E[1,e], Σ[1,e],
            sμ[n_p,e_left], sρ[n_p,e_left], sE[n_p,e_left], sΣ[n_p,e_left],
            sμ[1,e], sρ[1,e], sE[1,e], sΣ[1,e]; sγ=sγ)

        # Right face correction
        dsμ[n_p,e] -= invJ * (fstar_sμR - fμ[n_p,e]) / w[n_p]
        dsρ[n_p,e] -= invJ * (fstar_sρR - fρ[n_p,e]) / w[n_p]
        dsE[n_p,e] -= invJ * (fstar_sER - fE[n_p,e]) / w[n_p]

        # Left face correction
        dsμ[1,e] += invJ * (fstar_sμL - fμ[1,e]) / w[1]
        dsρ[1,e] += invJ * (fstar_sρL - fρ[1,e]) / w[1]
        dsE[1,e] += invJ * (fstar_sEL - fE[1,e]) / w[1]
    end

    return nothing
end

# ============================================================
# Elliptic sensitivity RHS — Form 1 (direct)
# ============================================================

"""
    compute_sensitivity_elliptic_rhs_form1!(b, μ, ρ, E, Σ,
        sμ, sρ, sE, basis, mesh, α)

Form 1 RHS for the sensitivity elliptic equation (adjoints.tex line 152):
  b = 4α u_x ŝ_u_x + ŝ_ρ Σ/ρ² - α ∂_x(ŝ_ρ/ρ² · ∂_xΣ)

The last term requires SIP-like DG treatment (diffusion of Σ with
coefficient κ = ŝ_ρ/ρ²).
"""
function compute_sensitivity_elliptic_rhs_form1!(b, μ, ρ, E, Σ,
        sμ, sρ, sE, basis, mesh, α)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w
    invJ = 1 / J
    η_pen = n_p^2  # penalty parameter

    fill!(b, zero(eltype(b)))

    # --- Volume contributions ---
    for e in 1:N_e
        for i in 1:n_p
            # Compute u_x and su_x at node i
            u_x  = zero(eltype(μ))
            su_x = zero(eltype(sμ))
            for j in 1:n_p
                u_j  = μ[j,e] / ρ[j,e]
                su_j = (sμ[j,e] - u_j * sρ[j,e]) / ρ[j,e]
                u_x  += D[i,j] * u_j
                su_x += D[i,j] * su_j
            end
            u_x  *= invJ
            su_x *= invJ

            # Pointwise source: 4α u_x su_x + sρ Σ / ρ²
            source = 4 * α * u_x * su_x + sρ[i,e] * Σ[i,e] / ρ[i,e]^2
            b[i,e] += w[i] * J * source

            # Diffusion SIP volume stiffness with κ = sρ/ρ²
            # α ∫ κ (∂_xΣ)(∂_x ψ_k) dx
            κ_i = sρ[i,e] / ρ[i,e]^2
            dxi_Σ = zero(eltype(Σ))
            for j in 1:n_p
                dxi_Σ += D[i,j] * Σ[j,e]
            end
            for k in 1:n_p
                b[k,e] += α * w[i] * invJ * κ_i * dxi_Σ * D[i,k]
            end
        end
    end

    # --- Face contributions (SIP terms for diffusion with κ = sρ/ρ²) ---
    for e in 1:N_e
        e_R = e == N_e ? 1 : e + 1

        # Values at the interface
        Σ_L = Σ[n_p, e]
        Σ_R = Σ[1, e_R]
        κ_L = sρ[n_p, e] / ρ[n_p, e]^2
        κ_R = sρ[1, e_R] / ρ[1, e_R]^2

        # Jump [[Σ]] = Σ_L - Σ_R
        jump_Σ = Σ_L - Σ_R

        # Reference derivatives of Σ at face nodes
        dxi_Σ_L = zero(eltype(Σ))
        dxi_Σ_R = zero(eltype(Σ))
        for j in 1:n_p
            dxi_Σ_L += D[n_p, j] * Σ[j, e]
            dxi_Σ_R += D[1, j] * Σ[j, e_R]
        end

        # Average flux: {{κ ∂Σ/∂x}}
        avg_flux = invJ * (κ_L * dxi_Σ_L + κ_R * dxi_Σ_R) / 2

        # Penalty coefficient
        κ_max = max(κ_L, κ_R)
        σ_pen = η_pen * κ_max / mesh.Δx

        # Consistency: -α {{κ ∂Σ/∂x}} [[ψ]]
        b[n_p, e]  += -α * avg_flux * (+1)
        b[1, e_R]  += -α * avg_flux * (-1)

        # Symmetry: -α {{κ ∂ψ/∂x}} [[Σ]]
        for k in 1:n_p
            b[k, e]    += -α * (κ_L / 2) * invJ * D[n_p, k] * (+1) * jump_Σ
            b[k, e_R]  += -α * (κ_R / 2) * invJ * D[1, k] * (+1) * jump_Σ
        end

        # Penalty: +α σ_pen [[Σ]] [[ψ]]
        b[n_p, e]  += α * σ_pen * jump_Σ
        b[1, e_R]  -= α * σ_pen * jump_Σ
    end

    return nothing
end

# ============================================================
# Elliptic sensitivity RHS — Form 2 (product-rule simplified)
# ============================================================

"""
    compute_sensitivity_elliptic_rhs_form2!(b, μ, ρ, E, Σ,
        sμ, sρ, sE, basis, mesh, α)

Form 2 RHS for the sensitivity elliptic equation (adjoints.tex line 163):
  b = α(2 u_x (2 ŝ_u_x + ŝ_ρ/ρ · u_x) - ∂_x(ŝ_ρ/ρ) · ∂_xΣ/ρ)

Only requires pointwise evaluation of first derivatives via D matrix.
No SIP face terms needed.
"""
function compute_sensitivity_elliptic_rhs_form2!(b, μ, ρ, E, Σ,
        sμ, sρ, sE, basis, mesh, α)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w
    invJ = 1 / J

    fill!(b, zero(eltype(b)))

    for e in 1:N_e
        for i in 1:n_p
            # Compute derivatives at node i via D matrix
            u_x     = zero(eltype(μ))
            su_x    = zero(eltype(sμ))
            dx_sρoρ = zero(eltype(sρ))  # ∂_x(sρ/ρ)
            dx_Σ    = zero(eltype(Σ))   # ∂_xΣ

            for j in 1:n_p
                u_j  = μ[j,e] / ρ[j,e]
                su_j = (sμ[j,e] - u_j * sρ[j,e]) / ρ[j,e]
                u_x     += D[i,j] * u_j
                su_x    += D[i,j] * su_j
                dx_sρoρ += D[i,j] * (sρ[j,e] / ρ[j,e])
                dx_Σ    += D[i,j] * Σ[j,e]
            end
            u_x     *= invJ
            su_x    *= invJ
            dx_sρoρ *= invJ
            dx_Σ    *= invJ

            # Pointwise RHS
            sρoρ_i = sρ[i,e] / ρ[i,e]
            R = α * (2 * u_x * (2 * su_x + sρoρ_i * u_x) -
                     dx_sρoρ * dx_Σ / ρ[i,e])

            b[i,e] = w[i] * J * R
        end
    end

    return nothing
end

# ============================================================
# Elliptic sensitivity solve
# ============================================================

"""
    solve_sensitivity_elliptic!(sΣ, Σ, μ, ρ, E, sμ, sρ, sE,
        basis, mesh, γ, α; n_iter=10, elliptic_rhs=:form2)

Solve the sensitivity elliptic equation for ŝ_Σ:
  A ŝ_Σ = b(form)
where A is the same SIP operator as the primal Σ equation.

`elliptic_rhs` selects the RHS form: `:form1` or `:form2`.
`sΣ` is used as initial guess (warm start).
"""
function solve_sensitivity_elliptic!(sΣ, Σ, μ, ρ, E, sμ, sρ, sE,
        basis, mesh, γ, α; n_iter=10, elliptic_rhs=:form2, solver=:pcg)
    b = similar(sΣ)

    if elliptic_rhs == :form1
        compute_sensitivity_elliptic_rhs_form1!(b, μ, ρ, E, Σ, sμ, sρ, sE, basis, mesh, α)
    elseif elliptic_rhs == :form2
        compute_sensitivity_elliptic_rhs_form2!(b, μ, ρ, E, Σ, sμ, sρ, sE, basis, mesh, α)
    else
        error("Unknown elliptic_rhs: $elliptic_rhs (use :form1 or :form2)")
    end

    # Solve with the same operator A as the primal
    M_diag = compute_sip_diagonal(ρ, basis, mesh, α)
    if solver == :pcg
        pcg_fixed!(sΣ, b, M_diag, n_iter, ρ, basis, mesh, α)
    elseif solver == :jacobi
        jacobi_fixed!(sΣ, b, M_diag, n_iter, ρ, basis, mesh, α, 2.0/3.0)
    elseif solver == :chebyshev
        chebyshev_fixed!(sΣ, b, M_diag, n_iter, ρ, basis, mesh, α)
    else
        error("Unknown solver: $solver (use :pcg, :jacobi, or :chebyshev)")
    end

    return nothing
end

# ============================================================
# Combined sensitivity RHS (elliptic + hyperbolic)
# ============================================================

"""
    compute_sensitivity_rhs!(dsμ, dsρ, dsE,
        μ, ρ, E, Σ, sμ, sρ, sE, sΣ, basis, mesh, γ, α;
        n_iter=10, elliptic_rhs=:form2, solver=:pcg, sγ=zero(γ))

Full sensitivity spatial operator:
1. Solve elliptic for sΣ (if α > 0)
2. Compute sensitivity hyperbolic RHS
"""
function compute_sensitivity_rhs!(dsμ, dsρ, dsE,
        μ, ρ, E, Σ, sμ, sρ, sE, sΣ, basis, mesh, γ, α;
        n_iter=10, elliptic_rhs=:form2, solver=:pcg, sγ=zero(γ))
    if α > 0
        solve_sensitivity_elliptic!(sΣ, Σ, μ, ρ, E, sμ, sρ, sE,
            basis, mesh, γ, α; n_iter=n_iter, elliptic_rhs=elliptic_rhs, solver=solver)
    else
        fill!(sΣ, zero(eltype(sΣ)))
    end

    compute_sensitivity_hyperbolic_rhs!(dsμ, dsρ, dsE,
        μ, ρ, E, Σ, sμ, sρ, sE, sΣ, basis, mesh, γ; sγ=sγ)

    return nothing
end

# ============================================================
# Coupled SSP-RK3 timestepper (forward + sensitivity)
# ============================================================

"""
    ssp_rk3_coupled(fwd_rhs!, sens_rhs!, μ0, ρ0, E0, sμ0, sρ0, sE0,
                    Δt, T, basis, mesh, γ; callback=nothing)

SSP-RK3 advancing both primal (μ,ρ,E) and sensitivity (sμ,sρ,sE).
At each stage, `fwd_rhs!` is called first (fills shared Σ),
then `sens_rhs!` (fills shared sΣ using the updated Σ).

Callback signature: callback(t, n_steps, μ, ρ, E, sμ, sρ, sE)
"""
function ssp_rk3_coupled(fwd_rhs!, sens_rhs!, μ0, ρ0, E0, sμ0, sρ0, sE0,
                         Δt, T, basis, mesh, γ; callback=nothing)
    μ  = copy(μ0);  ρ  = copy(ρ0);  E  = copy(E0)
    sμ = copy(sμ0); sρ = copy(sρ0); sE = copy(sE0)

    # Stage arrays — primal
    μ1 = similar(μ); ρ1 = similar(ρ); E1 = similar(E)
    μ2 = similar(μ); ρ2 = similar(ρ); E2 = similar(E)
    dμ = similar(μ); dρ = similar(ρ); dE = similar(E)

    # Stage arrays — sensitivity
    sμ1 = similar(sμ); sρ1 = similar(sρ); sE1 = similar(sE)
    sμ2 = similar(sμ); sρ2 = similar(sρ); sE2 = similar(sE)
    dsμ = similar(sμ); dsρ = similar(sρ); dsE = similar(sE)

    t = zero(eltype(Δt))
    n_steps = 0

    while t < T - 1e-14
        dt = min(Δt, T - t)

        # === Stage 1: q1 = q + dt * f(q) ===
        fwd_rhs!(dμ, dρ, dE, μ, ρ, E, basis, mesh, γ)
        sens_rhs!(dsμ, dsρ, dsE, μ, ρ, E, sμ, sρ, sE, basis, mesh, γ)

        @. μ1 = μ + dt * dμ;   ρ1 = ρ + dt * dρ;   E1 = E + dt * dE
        @. sμ1 = sμ + dt * dsμ; sρ1 = sρ + dt * dsρ; sE1 = sE + dt * dsE

        # === Stage 2: q2 = 3/4 q + 1/4 q1 + 1/4 dt * f(q1) ===
        fwd_rhs!(dμ, dρ, dE, μ1, ρ1, E1, basis, mesh, γ)
        sens_rhs!(dsμ, dsρ, dsE, μ1, ρ1, E1, sμ1, sρ1, sE1, basis, mesh, γ)

        @. μ2 = 3/4 * μ + 1/4 * μ1 + 1/4 * dt * dμ
        @. ρ2 = 3/4 * ρ + 1/4 * ρ1 + 1/4 * dt * dρ
        @. E2 = 3/4 * E + 1/4 * E1 + 1/4 * dt * dE
        @. sμ2 = 3/4 * sμ + 1/4 * sμ1 + 1/4 * dt * dsμ
        @. sρ2 = 3/4 * sρ + 1/4 * sρ1 + 1/4 * dt * dsρ
        @. sE2 = 3/4 * sE + 1/4 * sE1 + 1/4 * dt * dsE

        # === Stage 3: q_new = 1/3 q + 2/3 q2 + 2/3 dt * f(q2) ===
        fwd_rhs!(dμ, dρ, dE, μ2, ρ2, E2, basis, mesh, γ)
        sens_rhs!(dsμ, dsρ, dsE, μ2, ρ2, E2, sμ2, sρ2, sE2, basis, mesh, γ)

        @. μ = 1/3 * μ + 2/3 * μ2 + 2/3 * dt * dμ
        @. ρ = 1/3 * ρ + 2/3 * ρ2 + 2/3 * dt * dρ
        @. E = 1/3 * E + 2/3 * E2 + 2/3 * dt * dE
        @. sμ = 1/3 * sμ + 2/3 * sμ2 + 2/3 * dt * dsμ
        @. sρ = 1/3 * sρ + 2/3 * sρ2 + 2/3 * dt * dsρ
        @. sE = 1/3 * sE + 2/3 * sE2 + 2/3 * dt * dsE

        t += dt
        n_steps += 1

        if callback !== nothing
            callback(t, n_steps, μ, ρ, E, sμ, sρ, sE)
        end
    end

    return μ, ρ, E, sμ, sρ, sE
end

# ============================================================
# Driver: run_sensitivity
# ============================================================

"""
    run_sensitivity(μ0, ρ0, E0, sμ0, sρ0, sE0, Δt, T, basis, mesh, γ, α;
                    n_iter=10, elliptic_rhs=:form2, solver=:pcg, sγ=0.0, callback=nothing)

Run the coupled forward + sensitivity solve from t=0 to t=T.
Set sγ=1.0 to include the ∂P/∂γ source term for γ-sensitivity.

Returns (μf, ρf, Ef, Σf, sμf, sρf, sEf, sΣf) at final time.
"""
function run_sensitivity(μ0, ρ0, E0, sμ0, sρ0, sE0, Δt, T, basis, mesh, γ, α;
                         n_iter=10, elliptic_rhs=:form2, solver=:pcg, sγ=0.0, callback=nothing)
    # Promote types for consistency
    RT = promote_type(eltype(μ0), eltype(sμ0), typeof(α), typeof(γ), typeof(Δt), typeof(T))
    if RT !== eltype(μ0);  μ0  = RT.(μ0);  end
    if RT !== eltype(ρ0);  ρ0  = RT.(ρ0);  end
    if RT !== eltype(E0);  E0  = RT.(E0);  end
    if RT !== eltype(sμ0); sμ0 = RT.(sμ0); end
    if RT !== eltype(sρ0); sρ0 = RT.(sρ0); end
    if RT !== eltype(sE0); sE0 = RT.(sE0); end

    # Shared work arrays for entropic pressure and its sensitivity
    Σ  = zeros(RT, size(μ0))
    sΣ = zeros(RT, size(μ0))

    # Forward RHS closure (captures Σ, α, n_iter, solver)
    function fwd_rhs!(dμ, dρ, dE, μ, ρ, E, basis, mesh, γ)
        compute_igr_euler_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ, α;
                               n_iter=n_iter, solver=solver)
    end

    # Sensitivity RHS closure (captures Σ, sΣ, α, n_iter, elliptic_rhs, solver, sγ)
    function sens_rhs!(dsμ, dsρ, dsE, μ, ρ, E, sμ, sρ, sE, basis, mesh, γ)
        compute_sensitivity_rhs!(dsμ, dsρ, dsE,
            μ, ρ, E, Σ, sμ, sρ, sE, sΣ, basis, mesh, γ, α;
            n_iter=n_iter, elliptic_rhs=elliptic_rhs, solver=solver, sγ=sγ)
    end

    μf, ρf, Ef, sμf, sρf, sEf = ssp_rk3_coupled(
        fwd_rhs!, sens_rhs!, μ0, ρ0, E0, sμ0, sρ0, sE0,
        Δt, T, basis, mesh, γ; callback=callback)

    # Compute Σ and sΣ at final state
    if α > 0
        solve_elliptic!(Σ, μf, ρf, Ef, basis, mesh, α; n_iter=n_iter, solver=solver)
        solve_sensitivity_elliptic!(sΣ, Σ, μf, ρf, Ef, sμf, sρf, sEf,
            basis, mesh, γ, α; n_iter=n_iter, elliptic_rhs=elliptic_rhs, solver=solver)
    end

    return μf, ρf, Ef, Σ, sμf, sρf, sEf, sΣ
end
