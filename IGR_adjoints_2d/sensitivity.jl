# ============================================================
# Forward sensitivity equations for the 2D IGR Euler system.
#
# The sensitivity is the directional derivative of the forward
# solution with respect to a parameter θ:
#   ζx = ∂_θμx, ζy = ∂_θμy, σ = ∂_θρ, η = ∂_θE, ς = ∂_θΣ.
#
# Differentiating the IGR-Euler system (adjoints.tex eq.
# igr_euler) gives a linear PDE for (ζx,ζy,σ,η) with the same
# hyperbolic + elliptic structure as the primal, coupled one-way
# to the primal solution. These are the general-dimension forms
# of forward_sensitivities.tex (d=2), reducing to the 1D package
# in the d=1 limit.
#
# This module covers parameters θ that enter through the initial
# condition or forcing (sγ-type terms are not included).
# ============================================================

export sensitivity_flux_x, sensitivity_flux_y,
       sensitivity_llf_flux_x, sensitivity_llf_flux_y,
       compute_sensitivity_hyperbolic_rhs!,
       compute_sensitivity_elliptic_rhs!,
       solve_sensitivity_elliptic!,
       compute_sensitivity_rhs!,
       run_sensitivity

# ============================================================
# Pointwise sensitivity flux  (∂_θ of the physical Euler flux)
# ============================================================

# Pressure sensitivity for a parameter entering via IC/forcing
# (forward_sensitivities.tex eq. Phi_IC, d=2):
#   Φ = (γ-1)(η - σ|u|²/2 - ρ u·s)
@inline function _pressure_sensitivity(γ, ux, uy, ρ, σ, η, sux, suy)
    return (γ - 1) * (η - σ * (ux^2 + uy^2) / 2 - ρ * (ux * sux + uy * suy))
end

"""
    sensitivity_flux_x(γ, μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς)

x-component of the sensitivity flux. Returns (fζx, fζy, fσ, fη).
"""
function sensitivity_flux_x(γ, μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς)
    ux = μx / ρ; uy = μy / ρ
    sux = (ζx - ux * σ) / ρ
    suy = (ζy - uy * σ) / ρ
    P  = polytropic_pressure(γ, μx, μy, ρ, E)
    Φ  = _pressure_sensitivity(γ, ux, uy, ρ, σ, η, sux, suy)
    Pθ = Φ + ς

    fζx = ζx * ux + μx * sux + Pθ
    fζy = ζx * uy + μx * suy
    fσ  = ζx
    fη  = (η + Pθ) * ux + (E + P + Σ) * sux
    return (fζx, fζy, fσ, fη)
end

"""
    sensitivity_flux_y(γ, μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς)

y-component of the sensitivity flux. Returns (fζx, fζy, fσ, fη).
"""
function sensitivity_flux_y(γ, μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς)
    ux = μx / ρ; uy = μy / ρ
    sux = (ζx - ux * σ) / ρ
    suy = (ζy - uy * σ) / ρ
    P  = polytropic_pressure(γ, μx, μy, ρ, E)
    Φ  = _pressure_sensitivity(γ, ux, uy, ρ, σ, η, sux, suy)
    Pθ = Φ + ς

    fζx = ζx * uy + μx * suy
    fζy = ζy * uy + μy * suy + Pθ
    fσ  = ζy
    fη  = (η + Pθ) * uy + (E + P + Σ) * suy
    return (fζx, fζy, fσ, fη)
end

"""
    sensitivity_llf_flux_x(γ, primalL..., primalR..., sensL..., sensR...)

LLF numerical flux for the sensitivity system across an x-face. The
linearized system has the primal characteristic speeds, so the
dissipation uses the primal wavespeed with the forward sign.
"""
function sensitivity_llf_flux_x(γ, μxL, μyL, ρL, EL, ΣL,
                                   μxR, μyR, ρR, ER, ΣR,
                                   ζxL, ζyL, σL, ηL, ςL,
                                   ζxR, ζyR, σR, ηR, ςR)
    λ = max(max_wavespeed_x(γ, μxL, μyL, ρL, EL),
            max_wavespeed_x(γ, μxR, μyR, ρR, ER))
    fζxL, fζyL, fσL, fηL = sensitivity_flux_x(γ, μxL, μyL, ρL, EL, ΣL, ζxL, ζyL, σL, ηL, ςL)
    fζxR, fζyR, fσR, fηR = sensitivity_flux_x(γ, μxR, μyR, ρR, ER, ΣR, ζxR, ζyR, σR, ηR, ςR)
    fζx = (fζxL + fζxR)/2 + λ*(ζxL - ζxR)/2
    fζy = (fζyL + fζyR)/2 + λ*(ζyL - ζyR)/2
    fσ  = (fσL  + fσR) /2 + λ*(σL  - σR) /2
    fη  = (fηL  + fηR) /2 + λ*(ηL  - ηR) /2
    return (fζx, fζy, fσ, fη)
end

"""
    sensitivity_llf_flux_y(γ, primalL..., primalR..., sensL..., sensR...)

LLF numerical flux for the sensitivity system across a y-face.
"""
function sensitivity_llf_flux_y(γ, μxL, μyL, ρL, EL, ΣL,
                                   μxR, μyR, ρR, ER, ΣR,
                                   ζxL, ζyL, σL, ηL, ςL,
                                   ζxR, ζyR, σR, ηR, ςR)
    λ = max(max_wavespeed_y(γ, μxL, μyL, ρL, EL),
            max_wavespeed_y(γ, μxR, μyR, ρR, ER))
    fζxL, fζyL, fσL, fηL = sensitivity_flux_y(γ, μxL, μyL, ρL, EL, ΣL, ζxL, ζyL, σL, ηL, ςL)
    fζxR, fζyR, fσR, fηR = sensitivity_flux_y(γ, μxR, μyR, ρR, ER, ΣR, ζxR, ζyR, σR, ηR, ςR)
    fζx = (fζxL + fζxR)/2 + λ*(ζxL - ζxR)/2
    fζy = (fζyL + fζyR)/2 + λ*(ζyL - ζyR)/2
    fσ  = (fσL  + fσR) /2 + λ*(σL  - σR) /2
    fη  = (fηL  + fηR) /2 + λ*(ηL  - ηR) /2
    return (fζx, fζy, fσ, fη)
end

# ============================================================
# Sensitivity hyperbolic DG operator
# ============================================================

"""
    compute_sensitivity_hyperbolic_rhs!(dζx, dζy, dσ, dη,
        μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς, basis, mesh, γ)

DG right-hand side for the sensitivity hyperbolic equations.
Same strong-form tensor-product structure as the primal operator.
"""
function compute_sensitivity_hyperbolic_rhs!(dζx, dζy, dσ, dη,
        μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς, basis, mesh, γ)
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    w    = basis.w
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy

    fxζx = similar(ζx); fxζy = similar(ζx); fxσ = similar(ζx); fxη = similar(ζx)
    fyζx = similar(ζx); fyζy = similar(ζx); fyσ = similar(ζx); fyη = similar(ζx)
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            fxζx[i,j,ex,ey], fxζy[i,j,ex,ey], fxσ[i,j,ex,ey], fxη[i,j,ex,ey] =
                sensitivity_flux_x(γ, μx[i,j,ex,ey], μy[i,j,ex,ey], ρ[i,j,ex,ey],
                    E[i,j,ex,ey], Σ[i,j,ex,ey],
                    ζx[i,j,ex,ey], ζy[i,j,ex,ey], σ[i,j,ex,ey], η[i,j,ex,ey], ς[i,j,ex,ey])
            fyζx[i,j,ex,ey], fyζy[i,j,ex,ey], fyσ[i,j,ex,ey], fyη[i,j,ex,ey] =
                sensitivity_flux_y(γ, μx[i,j,ex,ey], μy[i,j,ex,ey], ρ[i,j,ex,ey],
                    E[i,j,ex,ey], Σ[i,j,ex,ey],
                    ζx[i,j,ex,ey], ζy[i,j,ex,ey], σ[i,j,ex,ey], η[i,j,ex,ey], ς[i,j,ex,ey])
        end
    end

    # Volume terms: dq = -(1/Jx) Dx·fx - (1/Jy) Dy·fy
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            aζx = zero(eltype(ζx)); aζy = zero(eltype(ζx))
            aσ  = zero(eltype(ζx)); aη  = zero(eltype(ζx))
            for k in 1:n_p
                aζx -= invJx*D[i,k]*fxζx[k,j,ex,ey] + invJy*D[j,k]*fyζx[i,k,ex,ey]
                aζy -= invJx*D[i,k]*fxζy[k,j,ex,ey] + invJy*D[j,k]*fyζy[i,k,ex,ey]
                aσ  -= invJx*D[i,k]*fxσ[k,j,ex,ey]  + invJy*D[j,k]*fyσ[i,k,ex,ey]
                aη  -= invJx*D[i,k]*fxη[k,j,ex,ey]  + invJy*D[j,k]*fyη[i,k,ex,ey]
            end
            dζx[i,j,ex,ey] = aζx; dζy[i,j,ex,ey] = aζy
            dσ[i,j,ex,ey]  = aσ;  dη[i,j,ex,ey]  = aη
        end
    end

    # x-normal faces
    for ey in 1:N_ey, ex in 1:N_ex
        exR = ex == N_ex ? 1 : ex + 1
        for j in 1:n_p
            fζx, fζy, fσ, fη = sensitivity_llf_flux_x(γ,
                μx[n_p,j,ex,ey], μy[n_p,j,ex,ey], ρ[n_p,j,ex,ey], E[n_p,j,ex,ey], Σ[n_p,j,ex,ey],
                μx[1,j,exR,ey],  μy[1,j,exR,ey],  ρ[1,j,exR,ey],  E[1,j,exR,ey],  Σ[1,j,exR,ey],
                ζx[n_p,j,ex,ey], ζy[n_p,j,ex,ey], σ[n_p,j,ex,ey], η[n_p,j,ex,ey], ς[n_p,j,ex,ey],
                ζx[1,j,exR,ey],  ζy[1,j,exR,ey],  σ[1,j,exR,ey],  η[1,j,exR,ey],  ς[1,j,exR,ey])

            dζx[n_p,j,ex,ey] -= invJx*(fζx - fxζx[n_p,j,ex,ey])/w[n_p]
            dζy[n_p,j,ex,ey] -= invJx*(fζy - fxζy[n_p,j,ex,ey])/w[n_p]
            dσ[n_p,j,ex,ey]  -= invJx*(fσ  - fxσ[n_p,j,ex,ey]) /w[n_p]
            dη[n_p,j,ex,ey]  -= invJx*(fη  - fxη[n_p,j,ex,ey]) /w[n_p]

            dζx[1,j,exR,ey] += invJx*(fζx - fxζx[1,j,exR,ey])/w[1]
            dζy[1,j,exR,ey] += invJx*(fζy - fxζy[1,j,exR,ey])/w[1]
            dσ[1,j,exR,ey]  += invJx*(fσ  - fxσ[1,j,exR,ey]) /w[1]
            dη[1,j,exR,ey]  += invJx*(fη  - fxη[1,j,exR,ey]) /w[1]
        end
    end

    # y-normal faces
    for ey in 1:N_ey, ex in 1:N_ex
        eyR = ey == N_ey ? 1 : ey + 1
        for i in 1:n_p
            fζx, fζy, fσ, fη = sensitivity_llf_flux_y(γ,
                μx[i,n_p,ex,ey], μy[i,n_p,ex,ey], ρ[i,n_p,ex,ey], E[i,n_p,ex,ey], Σ[i,n_p,ex,ey],
                μx[i,1,ex,eyR],  μy[i,1,ex,eyR],  ρ[i,1,ex,eyR],  E[i,1,ex,eyR],  Σ[i,1,ex,eyR],
                ζx[i,n_p,ex,ey], ζy[i,n_p,ex,ey], σ[i,n_p,ex,ey], η[i,n_p,ex,ey], ς[i,n_p,ex,ey],
                ζx[i,1,ex,eyR],  ζy[i,1,ex,eyR],  σ[i,1,ex,eyR],  η[i,1,ex,eyR],  ς[i,1,ex,eyR])

            dζx[i,n_p,ex,ey] -= invJy*(fζx - fyζx[i,n_p,ex,ey])/w[n_p]
            dζy[i,n_p,ex,ey] -= invJy*(fζy - fyζy[i,n_p,ex,ey])/w[n_p]
            dσ[i,n_p,ex,ey]  -= invJy*(fσ  - fyσ[i,n_p,ex,ey]) /w[n_p]
            dη[i,n_p,ex,ey]  -= invJy*(fη  - fyη[i,n_p,ex,ey]) /w[n_p]

            dζx[i,1,ex,eyR] += invJy*(fζx - fyζx[i,1,ex,eyR])/w[1]
            dζy[i,1,ex,eyR] += invJy*(fζy - fyζy[i,1,ex,eyR])/w[1]
            dσ[i,1,ex,eyR]  += invJy*(fσ  - fyσ[i,1,ex,eyR]) /w[1]
            dη[i,1,ex,eyR]  += invJy*(fη  - fyη[i,1,ex,eyR]) /w[1]
        end
    end
    return nothing
end

# ============================================================
# Sensitivity elliptic RHS (product-rule simplified form)
#
# Using the primal elliptic identity Σ/ρ = R - α div(∇Σ/ρ) to
# eliminate the divergence terms, the ς-equation RHS becomes the
# purely pointwise expression
#   b = R_sens + (σ/ρ) R_primal - α ∇(σ/ρ)·(∇Σ/ρ)
# with
#   R_primal = α[ (∇·u)² + tr((Du)²) ]
#   R_sens   = α[ 2(∇·u)(∇·s) + 2 Σ_ij u_{i,j} s_{j,i} ].
# ============================================================

"""
    compute_sensitivity_elliptic_rhs!(b, μx, μy, ρ, E, Σ,
        ζx, ζy, σ, η, basis, mesh, α)

Assemble the discrete RHS `b` of the sensitivity elliptic equation
for ς (no SIP face terms needed; see module header).
"""
function compute_sensitivity_elliptic_rhs!(b, μx, μy, ρ, E, Σ,
        ζx, ζy, σ, η, basis, mesh, α)
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    w    = basis.w
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy

    fill!(b, zero(eltype(b)))

    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            uxx = zero(eltype(μx)); uxy = zero(eltype(μx))
            uyx = zero(eltype(μx)); uyy = zero(eltype(μx))
            sxx = zero(eltype(μx)); sxy = zero(eltype(μx))
            syx = zero(eltype(μx)); syy = zero(eltype(μx))
            Σx  = zero(eltype(μx)); Σy  = zero(eltype(μx))
            dx_sρ = zero(eltype(μx)); dy_sρ = zero(eltype(μx))  # ∂(σ/ρ)
            for k in 1:n_p
                # primal velocity
                uxkj = μx[k,j,ex,ey]/ρ[k,j,ex,ey]
                uykj = μy[k,j,ex,ey]/ρ[k,j,ex,ey]
                uxik = μx[i,k,ex,ey]/ρ[i,k,ex,ey]
                uyik = μy[i,k,ex,ey]/ρ[i,k,ex,ey]
                # sensitivity velocity  su = (ζ - u σ)/ρ
                sxkj = (ζx[k,j,ex,ey] - uxkj*σ[k,j,ex,ey])/ρ[k,j,ex,ey]
                sykj = (ζy[k,j,ex,ey] - uykj*σ[k,j,ex,ey])/ρ[k,j,ex,ey]
                sxik = (ζx[i,k,ex,ey] - uxik*σ[i,k,ex,ey])/ρ[i,k,ex,ey]
                syik = (ζy[i,k,ex,ey] - uyik*σ[i,k,ex,ey])/ρ[i,k,ex,ey]
                uxx += D[i,k]*uxkj; uyx += D[i,k]*uykj
                uxy += D[j,k]*uxik; uyy += D[j,k]*uyik
                sxx += D[i,k]*sxkj; syx += D[i,k]*sykj
                sxy += D[j,k]*sxik; syy += D[j,k]*syik
                Σx  += D[i,k]*Σ[k,j,ex,ey]; Σy += D[j,k]*Σ[i,k,ex,ey]
                dx_sρ += D[i,k]*(σ[k,j,ex,ey]/ρ[k,j,ex,ey])
                dy_sρ += D[j,k]*(σ[i,k,ex,ey]/ρ[i,k,ex,ey])
            end
            uxx *= invJx; uyx *= invJx; uxy *= invJy; uyy *= invJy
            sxx *= invJx; syx *= invJx; sxy *= invJy; syy *= invJy
            Σx  *= invJx; Σy  *= invJy
            dx_sρ *= invJx; dy_sρ *= invJy

            divu = uxx + uyy
            divs = sxx + syy
            R_primal = α * (divu^2 + uxx^2 + 2*uxy*uyx + uyy^2)
            R_sens   = α * (2*divu*divs +
                            2*(uxx*sxx + uxy*syx + uyx*sxy + uyy*syy))

            r = ρ[i,j,ex,ey]
            sρ_i = σ[i,j,ex,ey] / r
            b_pt = R_sens + sρ_i * R_primal - α*(dx_sρ*Σx + dy_sρ*Σy)/r
            b[i,j,ex,ey] = w[i] * w[j] * mesh.Jx * mesh.Jy * b_pt
        end
    end
    return nothing
end

"""
    solve_sensitivity_elliptic!(ς, μx, μy, ρ, E, Σ, ζx, ζy, σ, η,
        basis, mesh, γ, α; n_iter=10, solver=:pcg)

Solve the sensitivity elliptic equation for ς. The operator is the
same SIP operator as the primal Σ equation; `ς` is the warm start.
"""
function solve_sensitivity_elliptic!(ς, μx, μy, ρ, E, Σ, ζx, ζy, σ, η,
        basis, mesh, γ, α; n_iter=10, solver=:pcg)
    b = similar(ς)
    compute_sensitivity_elliptic_rhs!(b, μx, μy, ρ, E, Σ, ζx, ζy, σ, η, basis, mesh, α)
    M_diag = compute_sip_diagonal(ρ, basis, mesh, α)
    if solver == :pcg
        pcg_fixed!(ς, b, M_diag, n_iter, ρ, basis, mesh, α)
    elseif solver == :jacobi
        jacobi_fixed!(ς, b, M_diag, n_iter, ρ, basis, mesh, α, 2.0/3.0)
    elseif solver == :chebyshev
        chebyshev_fixed!(ς, b, M_diag, n_iter, ρ, basis, mesh, α)
    else
        error("Unknown solver: $solver")
    end
    return nothing
end

"""
    compute_sensitivity_rhs!(dζx, dζy, dσ, dη, μx, μy, ρ, E, Σ,
        ζx, ζy, σ, η, ς, basis, mesh, γ, α; n_iter=10, solver=:pcg)

Full sensitivity spatial operator: solve the elliptic equation for ς
(if α > 0), then evaluate the sensitivity hyperbolic DG operator.
"""
function compute_sensitivity_rhs!(dζx, dζy, dσ, dη, μx, μy, ρ, E, Σ,
        ζx, ζy, σ, η, ς, basis, mesh, γ, α; n_iter=10, solver=:pcg)
    if α > 0
        solve_sensitivity_elliptic!(ς, μx, μy, ρ, E, Σ, ζx, ζy, σ, η,
            basis, mesh, γ, α; n_iter=n_iter, solver=solver)
    else
        fill!(ς, zero(eltype(ς)))
    end
    compute_sensitivity_hyperbolic_rhs!(dζx, dζy, dσ, dη,
        μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς, basis, mesh, γ)
    return nothing
end

# ============================================================
# Coupled forward + sensitivity time integration (SSP-RK3)
# ============================================================

"""
    run_sensitivity(μx0, μy0, ρ0, E0, ζx0, ζy0, σ0, η0, Δt, T,
                    basis, mesh, γ, α; n_iter=10, solver=:pcg)

Advance the primal IGR-Euler state and the forward sensitivities
together from t=0 to t=T with SSP-RK3. At each stage the primal RHS
is evaluated first (filling the shared Σ), then the sensitivity RHS
(filling the shared ς from the updated Σ).

Returns `(μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς)` at the final time.
"""
function run_sensitivity(μx0, μy0, ρ0, E0, ζx0, ζy0, σ0, η0, Δt, T,
                         basis, mesh, γ, α; n_iter=10, solver=:pcg)
    RT = promote_type(eltype(μx0), eltype(ζx0), typeof(α), typeof(γ),
                      typeof(Δt), typeof(T))
    μx = RT.(copy(μx0)); μy = RT.(copy(μy0)); ρ = RT.(copy(ρ0)); E = RT.(copy(E0))
    ζx = RT.(copy(ζx0)); ζy = RT.(copy(ζy0)); σ = RT.(copy(σ0)); η = RT.(copy(η0))
    Σ = zeros(RT, size(μx)); ς = zeros(RT, size(μx))

    # Primal stage arrays
    μx1 = similar(μx); μy1 = similar(μy); ρ1 = similar(ρ); E1 = similar(E)
    μx2 = similar(μx); μy2 = similar(μy); ρ2 = similar(ρ); E2 = similar(E)
    dμx = similar(μx); dμy = similar(μy); dρ = similar(ρ); dE = similar(E)
    # Sensitivity stage arrays
    ζx1 = similar(ζx); ζy1 = similar(ζy); σ1 = similar(σ); η1 = similar(η)
    ζx2 = similar(ζx); ζy2 = similar(ζy); σ2 = similar(σ); η2 = similar(η)
    dζx = similar(ζx); dζy = similar(ζy); dσ = similar(σ); dη = similar(η)

    n_steps = ceil(Int, T / Δt)
    Δt = T / n_steps

    for _ in 1:n_steps
        # Stage 1
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        compute_sensitivity_rhs!(dζx, dζy, dσ, dη, μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μx1 = μx + Δt*dμx; @. μy1 = μy + Δt*dμy
        @. ρ1  = ρ  + Δt*dρ;  @. E1  = E  + Δt*dE
        @. ζx1 = ζx + Δt*dζx; @. ζy1 = ζy + Δt*dζy
        @. σ1  = σ  + Δt*dσ;  @. η1  = η  + Δt*dη

        # Stage 2
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx1, μy1, ρ1, E1, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        compute_sensitivity_rhs!(dζx, dζy, dσ, dη, μx1, μy1, ρ1, E1, Σ, ζx1, ζy1, σ1, η1, ς, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μx2 = 3/4*μx + 1/4*μx1 + 1/4*Δt*dμx
        @. μy2 = 3/4*μy + 1/4*μy1 + 1/4*Δt*dμy
        @. ρ2  = 3/4*ρ  + 1/4*ρ1  + 1/4*Δt*dρ
        @. E2  = 3/4*E  + 1/4*E1  + 1/4*Δt*dE
        @. ζx2 = 3/4*ζx + 1/4*ζx1 + 1/4*Δt*dζx
        @. ζy2 = 3/4*ζy + 1/4*ζy1 + 1/4*Δt*dζy
        @. σ2  = 3/4*σ  + 1/4*σ1  + 1/4*Δt*dσ
        @. η2  = 3/4*η  + 1/4*η1  + 1/4*Δt*dη

        # Stage 3
        compute_igr_euler_rhs!(dμx, dμy, dρ, dE, μx2, μy2, ρ2, E2, Σ, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        compute_sensitivity_rhs!(dζx, dζy, dσ, dη, μx2, μy2, ρ2, E2, Σ, ζx2, ζy2, σ2, η2, ς, basis, mesh, γ, α; n_iter=n_iter, solver=solver)
        @. μx = 1/3*μx + 2/3*μx2 + 2/3*Δt*dμx
        @. μy = 1/3*μy + 2/3*μy2 + 2/3*Δt*dμy
        @. ρ  = 1/3*ρ  + 2/3*ρ2  + 2/3*Δt*dρ
        @. E  = 1/3*E  + 2/3*E2  + 2/3*Δt*dE
        @. ζx = 1/3*ζx + 2/3*ζx2 + 2/3*Δt*dζx
        @. ζy = 1/3*ζy + 2/3*ζy2 + 2/3*Δt*dζy
        @. σ  = 1/3*σ  + 2/3*σ2  + 2/3*Δt*dσ
        @. η  = 1/3*η  + 2/3*η2  + 2/3*Δt*dη
    end

    if α > 0
        solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α; n_iter=n_iter, solver=solver)
        solve_sensitivity_elliptic!(ς, μx, μy, ρ, E, Σ, ζx, ζy, σ, η,
            basis, mesh, γ, α; n_iter=n_iter, solver=solver)
    end
    return μx, μy, ρ, E, Σ, ζx, ζy, σ, η, ς
end
