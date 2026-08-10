# ============================================================
# Continuous adjoint PDE solver for the 2D IGR Euler system
# (optimize-then-discretize, conservative formulation).
#
# Reversing time (τ = T - t), the adjoint equation is a
# forward-in-τ conservation law with source:
#   ∂_τ q† = +div(F†_loc) - S_loc + S_Σ
# where F†_loc is the local adjoint flux (physical-pressure
# derivatives only, Σ frozen), S_loc the local source, and
# S_Σ the nonlocal source obtained from a single adjoint
# elliptic solve for Π.
#
# The 2D forms below are the general-dimension expressions of
# adjoints.tex (§"conservative form" and §"nonlocal source");
# they reduce exactly to the 1D package in the d=1 limit.
#
# Conserved/adjoint ordering: (μx, μy, ρ, E).
# ============================================================

export adjoint_flux_local_x, adjoint_flux_local_y,
       adjoint_llf_flux_x, adjoint_llf_flux_y,
       compute_adjoint_source_local!,
       compute_adjoint_hyperbolic_rhs_local!,
       compute_adjoint_elliptic_rhs!,
       compute_nonlocal_source!,
       compute_adjoint_rhs_conservative!,
       run_adjoint_conservative

# ============================================================
# Local adjoint flux F†_loc  (physical-pressure derivatives only)
#
# For adjoint component a ∈ {μx,μy,ρ,E}, F†_loc[a] is a 2-vector
# (x- and y-flux). Derived from ∇_q(q†·F) with Σ held frozen.
# ============================================================

"""
    adjoint_flux_local_x(γ, μx, μy, ρ, E, Σ, aμx, aμy, aρ, aE)

x-components of the local adjoint flux. Returns (fμx, fμy, fρ, fE).
"""
function adjoint_flux_local_x(γ, μx, μy, ρ, E, Σ, aμx, aμy, aρ, aE)
    ux = μx / ρ; uy = μy / ρ
    P  = polytropic_pressure(γ, μx, μy, ρ, E)
    H̄  = E + P + Σ
    Pμx = -(γ - 1) * ux
    Pμy = -(γ - 1) * uy
    Pρ  =  (γ - 1) * (ux^2 + uy^2) / 2
    PE  =   γ - 1
    aμ_μ = aμx * μx + aμy * μy   # μ†·μ

    fμx = aμ_μ / ρ + aμx * ux + Pμx * aμx + aρ + aE * (H̄/ρ) + aE * Pμx * μx / ρ
    fμy = aμy * ux + Pμy * aμx + aE * Pμy * μx / ρ
    fρ  = -ux * aμ_μ / ρ + Pρ * aμx + aE * μx * (Pρ/ρ - H̄/ρ^2)
    fE  = PE * aμx + aE * (1 + PE) * ux
    return (fμx, fμy, fρ, fE)
end

"""
    adjoint_flux_local_y(γ, μx, μy, ρ, E, Σ, aμx, aμy, aρ, aE)

y-components of the local adjoint flux. Returns (fμx, fμy, fρ, fE).
"""
function adjoint_flux_local_y(γ, μx, μy, ρ, E, Σ, aμx, aμy, aρ, aE)
    ux = μx / ρ; uy = μy / ρ
    P  = polytropic_pressure(γ, μx, μy, ρ, E)
    H̄  = E + P + Σ
    Pμx = -(γ - 1) * ux
    Pμy = -(γ - 1) * uy
    Pρ  =  (γ - 1) * (ux^2 + uy^2) / 2
    PE  =   γ - 1
    aμ_μ = aμx * μx + aμy * μy

    fμx = aμx * uy + Pμx * aμy + aE * Pμx * μy / ρ
    fμy = aμ_μ / ρ + aμy * uy + Pμy * aμy + aρ + aE * (H̄/ρ) + aE * Pμy * μy / ρ
    fρ  = -uy * aμ_μ / ρ + Pρ * aμy + aE * μy * (Pρ/ρ - H̄/ρ^2)
    fE  = PE * aμy + aE * (1 + PE) * uy
    return (fμx, fμy, fρ, fE)
end

"""
    adjoint_llf_flux_x(γ, primalL..., primalR..., adjointL..., adjointR...)

LLF numerical flux for the local adjoint system across an x-face.
The adjoint dissipation has the opposite sign of the forward flux
(the adjoint advects backward in time). Returns (fμx, fμy, fρ, fE).
"""
function adjoint_llf_flux_x(γ, μxL, μyL, ρL, EL, ΣL,
                               μxR, μyR, ρR, ER, ΣR,
                               aμxL, aμyL, aρL, aEL,
                               aμxR, aμyR, aρR, aER)
    λ = max(max_wavespeed_x(γ, μxL, μyL, ρL, EL),
            max_wavespeed_x(γ, μxR, μyR, ρR, ER))
    fμxL, fμyL, fρL, fEL = adjoint_flux_local_x(γ, μxL, μyL, ρL, EL, ΣL, aμxL, aμyL, aρL, aEL)
    fμxR, fμyR, fρR, fER = adjoint_flux_local_x(γ, μxR, μyR, ρR, ER, ΣR, aμxR, aμyR, aρR, aER)
    fμx = (fμxL + fμxR)/2 - λ*(aμxL - aμxR)/2
    fμy = (fμyL + fμyR)/2 - λ*(aμyL - aμyR)/2
    fρ  = (fρL  + fρR) /2 - λ*(aρL  - aρR) /2
    fE  = (fEL  + fER) /2 - λ*(aEL  - aER) /2
    return (fμx, fμy, fρ, fE)
end

"""
    adjoint_llf_flux_y(γ, primalL..., primalR..., adjointL..., adjointR...)

LLF numerical flux for the local adjoint system across a y-face.
"""
function adjoint_llf_flux_y(γ, μxL, μyL, ρL, EL, ΣL,
                               μxR, μyR, ρR, ER, ΣR,
                               aμxL, aμyL, aρL, aEL,
                               aμxR, aμyR, aρR, aER)
    λ = max(max_wavespeed_y(γ, μxL, μyL, ρL, EL),
            max_wavespeed_y(γ, μxR, μyR, ρR, ER))
    fμxL, fμyL, fρL, fEL = adjoint_flux_local_y(γ, μxL, μyL, ρL, EL, ΣL, aμxL, aμyL, aρL, aEL)
    fμxR, fμyR, fρR, fER = adjoint_flux_local_y(γ, μxR, μyR, ρR, ER, ΣR, aμxR, aμyR, aρR, aER)
    fμx = (fμxL + fμxR)/2 - λ*(aμxL - aμxR)/2
    fμy = (fμyL + fμyR)/2 - λ*(aμyL - aμyR)/2
    fρ  = (fρL  + fρR) /2 - λ*(aρL  - aρR) /2
    fE  = (fEL  + fER) /2 - λ*(aEL  - aER) /2
    return (fμx, fμy, fρ, fE)
end

# ============================================================
# Local adjoint source S_loc
#
# S_loc[k] = Σ_l a_l · ∂(divF_l)/∂q_k  (pointwise partials, the
# physical flux divergence differentiated w.r.t. the pointwise
# state with the spatial-derivative quantities held fixed).
# ============================================================

"""
    compute_adjoint_source_local!(Sμx, Sμy, Sρ, SE, aμx, aμy, aρ, aE,
                                  μx, μy, ρ, E, Σ, basis, mesh, γ)

Compute the local adjoint source (physical-pressure derivatives only).
"""
function compute_adjoint_source_local!(Sμx, Sμy, Sρ, SE, aμx, aμy, aρ, aE,
                                       μx, μy, ρ, E, Σ, basis, mesh, γ)
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy
    γm = γ - 1

    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            # Spatial derivatives of the primal state at node (i,j)
            mxx = zero(eltype(μx)); mxy = zero(eltype(μx))
            myx = zero(eltype(μx)); myy = zero(eltype(μx))
            rx  = zero(eltype(μx)); ry  = zero(eltype(μx))
            Ex  = zero(eltype(μx)); Ey  = zero(eltype(μx))
            Sx  = zero(eltype(μx)); Sy  = zero(eltype(μx))
            for k in 1:n_p
                mxx += D[i,k]*μx[k,j,ex,ey]; mxy += D[j,k]*μx[i,k,ex,ey]
                myx += D[i,k]*μy[k,j,ex,ey]; myy += D[j,k]*μy[i,k,ex,ey]
                rx  += D[i,k]*ρ[k,j,ex,ey];  ry  += D[j,k]*ρ[i,k,ex,ey]
                Ex  += D[i,k]*E[k,j,ex,ey];  Ey  += D[j,k]*E[i,k,ex,ey]
                Sx  += D[i,k]*Σ[k,j,ex,ey];  Sy  += D[j,k]*Σ[i,k,ex,ey]
            end
            mxx *= invJx; myx *= invJx; rx *= invJx; Ex *= invJx; Sx *= invJx
            mxy *= invJy; myy *= invJy; ry *= invJy; Ey *= invJy; Sy *= invJy

            r  = ρ[i,j,ex,ey]; mx = μx[i,j,ex,ey]; my = μy[i,j,ex,ey]
            en = E[i,j,ex,ey]
            ux = mx/r; uy = my/r
            uxx = (mxx - ux*rx)/r; uxy = (mxy - ux*ry)/r
            uyx = (myx - uy*rx)/r; uyy = (myy - uy*ry)/r

            P  = γm * (en - (mx^2 + my^2)/(2*r))
            H̄  = en + P + Σ[i,j,ex,ey]
            Pρ = γm * (ux^2 + uy^2) / 2

            # ∂_x P, ∂_y P and their pointwise q-derivatives (Σ frozen)
            P̄x = γm*(Ex - (mx*mxx + my*myx)/r + (mx^2+my^2)*rx/(2*r^2)) + Sx
            P̄y = γm*(Ey - (mx*mxy + my*myy)/r + (mx^2+my^2)*ry/(2*r^2)) + Sy
            dPx_dμx = γm*(-mxx/r + mx*rx/r^2)
            dPx_dμy = γm*(-myx/r + my*rx/r^2)
            dPx_dρ  = γm*((mx*mxx + my*myx)/r^2 - (mx^2+my^2)*rx/r^3)
            dPy_dμx = γm*(-mxy/r + mx*ry/r^2)
            dPy_dμy = γm*(-myy/r + my*ry/r^2)
            dPy_dρ  = γm*((mx*mxy + my*myy)/r^2 - (mx^2+my^2)*ry/r^3)

            HEx = Ex + P̄x; HEy = Ey + P̄y

            # ∂(divF_μx)/∂q
            dμx_μx = mxx/r + uxx - mx*rx/r^2 + dPx_dμx + uyy
            dμx_μy = dPx_dμy + mxy/r - mx*ry/r^2
            dμx_ρ  = mxx*(-ux/r) + mx*(ux*rx/r^2 - uxx/r) + dPx_dρ +
                     mxy*(-uy/r) + mx*(uy*ry/r^2 - uyy/r)
            # ∂(divF_μy)/∂q
            dμy_μx = myx/r - my*rx/r^2 + dPy_dμx
            dμy_μy = uxx + myy/r + uyy - my*ry/r^2 + dPy_dμy
            dμy_ρ  = myx*(-ux/r) + my*(ux*rx/r^2 - uxx/r) +
                     myy*(-uy/r) + my*(uy*ry/r^2 - uyy/r) + dPy_dρ
            # ∂(divF_E)/∂q
            dE_μx = dPx_dμx*ux + HEx/r + (-γm*ux)*uxx + H̄*(-rx/r^2) +
                    dPy_dμx*uy + (-γm*ux)*uyy
            dE_μy = dPx_dμy*ux + (-γm*uy)*uxx + dPy_dμy*uy + HEy/r +
                    (-γm*uy)*uyy + H̄*(-ry/r^2)
            dE_ρ  = dPx_dρ*ux + HEx*(-ux/r) + Pρ*uxx + H̄*(ux*rx/r^2 - uxx/r) +
                    dPy_dρ*uy + HEy*(-uy/r) + Pρ*uyy + H̄*(uy*ry/r^2 - uyy/r)
            dE_E  = γ * (uxx + uyy)

            axx = aμx[i,j,ex,ey]; ayy = aμy[i,j,ex,ey]; aEE = aE[i,j,ex,ey]
            Sμx[i,j,ex,ey] = axx*dμx_μx + ayy*dμy_μx + aEE*dE_μx
            Sμy[i,j,ex,ey] = axx*dμx_μy + ayy*dμy_μy + aEE*dE_μy
            Sρ[i,j,ex,ey]  = axx*dμx_ρ  + ayy*dμy_ρ  + aEE*dE_ρ
            SE[i,j,ex,ey]  = aEE*dE_E
        end
    end
    return nothing
end

# ============================================================
# Local adjoint DG operator: +div(F†_loc) - S_loc
# ============================================================

"""
    compute_adjoint_hyperbolic_rhs_local!(daμx, daμy, daρ, daE,
        aμx, aμy, aρ, aE, μx, μy, ρ, E, Σ, basis, mesh, γ)

DG right-hand side of the local adjoint equation. Uses the strong
form with adjoint LLF interface fluxes; signs of the volume and face
terms are flipped relative to the forward operator (backward advection).
"""
function compute_adjoint_hyperbolic_rhs_local!(daμx, daμy, daρ, daE,
        aμx, aμy, aρ, aE, μx, μy, ρ, E, Σ, basis, mesh, γ)
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    w    = basis.w
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy

    # Local adjoint flux at all nodes (x- and y-directional)
    fxμx = similar(aμx); fxμy = similar(aμx); fxρ = similar(aμx); fxE = similar(aμx)
    fyμx = similar(aμx); fyμy = similar(aμx); fyρ = similar(aμx); fyE = similar(aμx)
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            fxμx[i,j,ex,ey], fxμy[i,j,ex,ey], fxρ[i,j,ex,ey], fxE[i,j,ex,ey] =
                adjoint_flux_local_x(γ, μx[i,j,ex,ey], μy[i,j,ex,ey], ρ[i,j,ex,ey],
                    E[i,j,ex,ey], Σ[i,j,ex,ey],
                    aμx[i,j,ex,ey], aμy[i,j,ex,ey], aρ[i,j,ex,ey], aE[i,j,ex,ey])
            fyμx[i,j,ex,ey], fyμy[i,j,ex,ey], fyρ[i,j,ex,ey], fyE[i,j,ex,ey] =
                adjoint_flux_local_y(γ, μx[i,j,ex,ey], μy[i,j,ex,ey], ρ[i,j,ex,ey],
                    E[i,j,ex,ey], Σ[i,j,ex,ey],
                    aμx[i,j,ex,ey], aμy[i,j,ex,ey], aρ[i,j,ex,ey], aE[i,j,ex,ey])
        end
    end

    # Volume terms: dq† = +(1/Jx) Dx·f†x + (1/Jy) Dy·f†y
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            aμx_ = zero(eltype(aμx)); aμy_ = zero(eltype(aμx))
            aρ_  = zero(eltype(aμx)); aE_  = zero(eltype(aμx))
            for k in 1:n_p
                aμx_ += invJx*D[i,k]*fxμx[k,j,ex,ey] + invJy*D[j,k]*fyμx[i,k,ex,ey]
                aμy_ += invJx*D[i,k]*fxμy[k,j,ex,ey] + invJy*D[j,k]*fyμy[i,k,ex,ey]
                aρ_  += invJx*D[i,k]*fxρ[k,j,ex,ey]  + invJy*D[j,k]*fyρ[i,k,ex,ey]
                aE_  += invJx*D[i,k]*fxE[k,j,ex,ey]  + invJy*D[j,k]*fyE[i,k,ex,ey]
            end
            daμx[i,j,ex,ey] = aμx_; daμy[i,j,ex,ey] = aμy_
            daρ[i,j,ex,ey]  = aρ_;  daE[i,j,ex,ey]  = aE_
        end
    end

    # x-normal faces (adjoint LLF; face signs flipped vs. forward)
    for ey in 1:N_ey, ex in 1:N_ex
        exR = ex == N_ex ? 1 : ex + 1
        for j in 1:n_p
            fμx, fμy, fρ, fE = adjoint_llf_flux_x(γ,
                μx[n_p,j,ex,ey], μy[n_p,j,ex,ey], ρ[n_p,j,ex,ey], E[n_p,j,ex,ey], Σ[n_p,j,ex,ey],
                μx[1,j,exR,ey],  μy[1,j,exR,ey],  ρ[1,j,exR,ey],  E[1,j,exR,ey],  Σ[1,j,exR,ey],
                aμx[n_p,j,ex,ey], aμy[n_p,j,ex,ey], aρ[n_p,j,ex,ey], aE[n_p,j,ex,ey],
                aμx[1,j,exR,ey],  aμy[1,j,exR,ey],  aρ[1,j,exR,ey],  aE[1,j,exR,ey])

            daμx[n_p,j,ex,ey] += invJx*(fμx - fxμx[n_p,j,ex,ey])/w[n_p]
            daμy[n_p,j,ex,ey] += invJx*(fμy - fxμy[n_p,j,ex,ey])/w[n_p]
            daρ[n_p,j,ex,ey]  += invJx*(fρ  - fxρ[n_p,j,ex,ey]) /w[n_p]
            daE[n_p,j,ex,ey]  += invJx*(fE  - fxE[n_p,j,ex,ey]) /w[n_p]

            daμx[1,j,exR,ey] -= invJx*(fμx - fxμx[1,j,exR,ey])/w[1]
            daμy[1,j,exR,ey] -= invJx*(fμy - fxμy[1,j,exR,ey])/w[1]
            daρ[1,j,exR,ey]  -= invJx*(fρ  - fxρ[1,j,exR,ey]) /w[1]
            daE[1,j,exR,ey]  -= invJx*(fE  - fxE[1,j,exR,ey]) /w[1]
        end
    end

    # y-normal faces
    for ey in 1:N_ey, ex in 1:N_ex
        eyR = ey == N_ey ? 1 : ey + 1
        for i in 1:n_p
            fμx, fμy, fρ, fE = adjoint_llf_flux_y(γ,
                μx[i,n_p,ex,ey], μy[i,n_p,ex,ey], ρ[i,n_p,ex,ey], E[i,n_p,ex,ey], Σ[i,n_p,ex,ey],
                μx[i,1,ex,eyR],  μy[i,1,ex,eyR],  ρ[i,1,ex,eyR],  E[i,1,ex,eyR],  Σ[i,1,ex,eyR],
                aμx[i,n_p,ex,ey], aμy[i,n_p,ex,ey], aρ[i,n_p,ex,ey], aE[i,n_p,ex,ey],
                aμx[i,1,ex,eyR],  aμy[i,1,ex,eyR],  aρ[i,1,ex,eyR],  aE[i,1,ex,eyR])

            daμx[i,n_p,ex,ey] += invJy*(fμx - fyμx[i,n_p,ex,ey])/w[n_p]
            daμy[i,n_p,ex,ey] += invJy*(fμy - fyμy[i,n_p,ex,ey])/w[n_p]
            daρ[i,n_p,ex,ey]  += invJy*(fρ  - fyρ[i,n_p,ex,ey]) /w[n_p]
            daE[i,n_p,ex,ey]  += invJy*(fE  - fyE[i,n_p,ex,ey]) /w[n_p]

            daμx[i,1,ex,eyR] -= invJy*(fμx - fyμx[i,1,ex,eyR])/w[1]
            daμy[i,1,ex,eyR] -= invJy*(fμy - fyμy[i,1,ex,eyR])/w[1]
            daρ[i,1,ex,eyR]  -= invJy*(fρ  - fyρ[i,1,ex,eyR]) /w[1]
            daE[i,1,ex,eyR]  -= invJy*(fE  - fyE[i,1,ex,eyR]) /w[1]
        end
    end

    # Subtract the local source
    Sμx = similar(aμx); Sμy = similar(aμx); Sρ = similar(aμx); SE = similar(aμx)
    compute_adjoint_source_local!(Sμx, Sμy, Sρ, SE, aμx, aμy, aρ, aE,
        μx, μy, ρ, E, Σ, basis, mesh, γ)
    @. daμx -= Sμx
    @. daμy -= Sμy
    @. daρ  -= Sρ
    @. daE  -= SE
    return nothing
end

# ============================================================
# RHS for the adjoint elliptic solve (Π equation, eq. for Π)
#   L[Π] = α (div(μ†) + ∇(E†)·u)
# ============================================================

"""
    compute_adjoint_elliptic_rhs!(b, aμx, aμy, aE, μx, μy, ρ, basis, mesh)

Compute the integrand `div(aμ) + ∇(aE)·u` of the adjoint elliptic RHS.
"""
function compute_adjoint_elliptic_rhs!(b, aμx, aμy, aE, μx, μy, ρ, basis, mesh)
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy

    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            daμx_dx = zero(eltype(aμx)); daμy_dy = zero(eltype(aμx))
            daE_dx  = zero(eltype(aμx)); daE_dy  = zero(eltype(aμx))
            for k in 1:n_p
                daμx_dx += D[i,k]*aμx[k,j,ex,ey]
                daμy_dy += D[j,k]*aμy[i,k,ex,ey]
                daE_dx  += D[i,k]*aE[k,j,ex,ey]
                daE_dy  += D[j,k]*aE[i,k,ex,ey]
            end
            daμx_dx *= invJx; daE_dx *= invJx
            daμy_dy *= invJy; daE_dy *= invJy
            ux = μx[i,j,ex,ey]/ρ[i,j,ex,ey]
            uy = μy[i,j,ex,ey]/ρ[i,j,ex,ey]
            b[i,j,ex,ey] = daμx_dx + daμy_dy + daE_dx*ux + daE_dy*uy
        end
    end
    return nothing
end

# ============================================================
# Nonlocal source S_Σ from the adjoint elliptic variable Π
# ============================================================

"""
    compute_nonlocal_source!(Sμx, Sμy, Sρ, SE, Π, μx, μy, ρ, Σ, basis, mesh)

Compute the nonlocal source S_Σ (general-dimension form, d=2):
  W   = ∇(Π div u) + div(Π [Du]ᵀ)
  S_μ = -2 W / ρ
  S_ρ = [Π·R_tr + div(Π ∇Σ/ρ)] / ρ + 2 (W·μ) / ρ²
  S_E = 0
with R_tr = tr²(Du) + tr((Du)²). Requires double differentiation.
"""
function compute_nonlocal_source!(Sμx, Sμy, Sρ, SE, Π, μx, μy, ρ, Σ, basis, mesh)
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy

    # Velocity gradients are formed by differentiating the *nodal* velocity
    # field u = μ/ρ ∈ V_h^p — the same discretization the forward IGR
    # operator (compute_igr_rhs!) uses to build R(q). The continuous adjoint
    # must be consistent with the forward operator it differentiates, so the
    # quotient rule on ∂μ, ∂ρ (which gives a different O(hᵖ) result) is wrong here.
    ux = μx ./ ρ
    uy = μy ./ ρ

    # First pass: velocity gradients, products with Π
    Pdivu = similar(Π)               # Π·(div u)
    Pxx = similar(Π); Pyx = similar(Π)  # Π·ux_x , Π·uy_x  (for W_x)
    Pxy = similar(Π); Pyy = similar(Π)  # Π·ux_y , Π·uy_y  (for W_y)
    Cx  = similar(Π); Cy  = similar(Π)  # Π·Σ_x/ρ , Π·Σ_y/ρ
    Rtr = similar(Π)                 # tr²(Du)+tr((Du)²)

    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            uxx = zero(eltype(Π)); uxy = zero(eltype(Π))
            uyx = zero(eltype(Π)); uyy = zero(eltype(Π))
            Σx  = zero(eltype(Π)); Σy  = zero(eltype(Π))
            for k in 1:n_p
                uxx += D[i,k]*ux[k,j,ex,ey]; uyx += D[i,k]*uy[k,j,ex,ey]
                uxy += D[j,k]*ux[i,k,ex,ey]; uyy += D[j,k]*uy[i,k,ex,ey]
                Σx  += D[i,k]*Σ[k,j,ex,ey];  Σy  += D[j,k]*Σ[i,k,ex,ey]
            end
            uxx *= invJx; uyx *= invJx; Σx *= invJx
            uxy *= invJy; uyy *= invJy; Σy *= invJy
            divu = uxx + uyy

            r    = ρ[i,j,ex,ey]
            Π_ij = Π[i,j,ex,ey]
            Pdivu[i,j,ex,ey] = Π_ij*divu
            Pxx[i,j,ex,ey] = Π_ij*uxx; Pyx[i,j,ex,ey] = Π_ij*uyx
            Pxy[i,j,ex,ey] = Π_ij*uxy; Pyy[i,j,ex,ey] = Π_ij*uyy
            Cx[i,j,ex,ey]  = Π_ij*Σx/r; Cy[i,j,ex,ey] = Π_ij*Σy/r
            Rtr[i,j,ex,ey] = divu^2 + uxx^2 + 2*uxy*uyx + uyy^2
        end
    end

    # Second pass: differentiate the products and assemble S_Σ
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            dx_Pdivu = zero(eltype(Π)); dy_Pdivu = zero(eltype(Π))
            dx_Pxx = zero(eltype(Π)); dy_Pyx = zero(eltype(Π))
            dx_Pxy = zero(eltype(Π)); dy_Pyy = zero(eltype(Π))
            dx_Cx  = zero(eltype(Π)); dy_Cy  = zero(eltype(Π))
            for k in 1:n_p
                dx_Pdivu += D[i,k]*Pdivu[k,j,ex,ey]
                dy_Pdivu += D[j,k]*Pdivu[i,k,ex,ey]
                dx_Pxx   += D[i,k]*Pxx[k,j,ex,ey]
                dy_Pyx   += D[j,k]*Pyx[i,k,ex,ey]
                dx_Pxy   += D[i,k]*Pxy[k,j,ex,ey]
                dy_Pyy   += D[j,k]*Pyy[i,k,ex,ey]
                dx_Cx    += D[i,k]*Cx[k,j,ex,ey]
                dy_Cy    += D[j,k]*Cy[i,k,ex,ey]
            end
            dx_Pdivu *= invJx; dy_Pdivu *= invJy
            dx_Pxx *= invJx; dy_Pyx *= invJy
            dx_Pxy *= invJx; dy_Pyy *= invJy
            dx_Cx  *= invJx; dy_Cy  *= invJy

            # W = ∇(Π div u) + div(Π [Du]ᵀ)
            Wx = dx_Pdivu + dx_Pxx + dy_Pyx
            Wy = dy_Pdivu + dx_Pxy + dy_Pyy
            divC = dx_Cx + dy_Cy

            r  = ρ[i,j,ex,ey]
            mx = μx[i,j,ex,ey]; my = μy[i,j,ex,ey]

            Sμx[i,j,ex,ey] = -2*Wx/r
            Sμy[i,j,ex,ey] = -2*Wy/r
            Sρ[i,j,ex,ey]  = (Π[i,j,ex,ey]*Rtr[i,j,ex,ey] + divC)/r +
                             2*(Wx*mx + Wy*my)/r^2
            SE[i,j,ex,ey]  = zero(eltype(Π))
        end
    end
    return nothing
end

# ============================================================
# Full conservative adjoint RHS (local + Π nonlocal source)
# ============================================================

"""
    compute_adjoint_rhs_conservative!(daμx, daμy, daρ, daE,
        aμx, aμy, aρ, aE, μx, μy, ρ, E, Σ, Π, basis, mesh, γ, α;
        n_iter=10, solver=:pcg)

Full adjoint spatial operator: local DG operator plus, if α > 0, the
nonlocal source obtained from one adjoint elliptic solve for Π.
"""
function compute_adjoint_rhs_conservative!(daμx, daμy, daρ, daE,
        aμx, aμy, aρ, aE, μx, μy, ρ, E, Σ, Π, basis, mesh, γ, α;
        n_iter=10, solver=:pcg)
    compute_adjoint_hyperbolic_rhs_local!(daμx, daμy, daρ, daE,
        aμx, aμy, aρ, aE, μx, μy, ρ, E, Σ, basis, mesh, γ)

    if α > 0
        N_ex = mesh.N_ex; N_ey = mesh.N_ey
        n_p  = basis.p + 1
        w    = basis.w

        b_Pi = similar(Π)
        compute_adjoint_elliptic_rhs!(b_Pi, aμx, aμy, aE, μx, μy, ρ, basis, mesh)
        for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
            b_Pi[i,j,ex,ey] *= α * w[i] * w[j] * mesh.Jx * mesh.Jy
        end

        M_diag = compute_sip_diagonal(ρ, basis, mesh, α)
        if solver == :pcg
            pcg_fixed!(Π, b_Pi, M_diag, n_iter, ρ, basis, mesh, α)
        elseif solver == :jacobi
            jacobi_fixed!(Π, b_Pi, M_diag, n_iter, ρ, basis, mesh, α, 2.0/3.0)
        elseif solver == :chebyshev
            chebyshev_fixed!(Π, b_Pi, M_diag, n_iter, ρ, basis, mesh, α)
        end

        Sμx = similar(daμx); Sμy = similar(daμy)
        Sρ  = similar(daρ);  SE  = similar(daE)
        compute_nonlocal_source!(Sμx, Sμy, Sρ, SE, Π, μx, μy, ρ, Σ, basis, mesh)
        @. daμx += Sμx
        @. daμy += Sμy
        @. daρ  += Sρ
        @. daE  += SE
    end
    return nothing
end

# ============================================================
# Backward-in-time adjoint integration
# ============================================================

"""
    run_adjoint_conservative(snapshots, n_steps, Δt, T, basis, mesh, γ, α;
                             n_iter=10, objective=:l2_density, solver=:pcg)

Integrate the adjoint PDE backward in time (forward in τ = T - t) with
SSP-RK3, using the conservative Π formulation. The terminal condition
is the L² gradient of the objective at t = T.

Returns the adjoint fields `(aμx, aμy, aρ, aE)` at t = 0.
"""
function run_adjoint_conservative(snapshots, n_steps, Δt, T, basis, mesh, γ, α;
                                  n_iter=10, objective=:l2_density, solver=:pcg)
    n_p  = basis.p + 1
    N_ex = mesh.N_ex; N_ey = mesh.N_ey

    μxT, μyT, ρT, ET, _ = snapshots[end]
    RT = eltype(ρT)

    aμx = zeros(RT, n_p, n_p, N_ex, N_ey)
    aμy = zeros(RT, n_p, n_p, N_ex, N_ey)
    aρ  = zeros(RT, n_p, n_p, N_ex, N_ey)
    aE  = zeros(RT, n_p, n_p, N_ex, N_ey)

    # Terminal condition: L² gradient of J at t = T
    if objective == :l2_density
        @. aρ = 2 * ρT
    elseif objective == :l2_pressure
        for idx in eachindex(ρT)
            P = (γ-1) * (ET[idx] - (μxT[idx]^2 + μyT[idx]^2)/(2*ρT[idx]))
            ux = μxT[idx]/ρT[idx]; uy = μyT[idx]/ρT[idx]
            aμx[idx] = -2*(γ-1)*P*ux
            aμy[idx] = -2*(γ-1)*P*uy
            aρ[idx]  =  (γ-1)*P*(ux^2 + uy^2)
            aE[idx]  =  2*(γ-1)*P
        end
    elseif objective == :weighted_momentum
        for idx in eachindex(ρT)
            g = sin(2π*mesh.x[idx]/mesh.Lx) * sin(2π*mesh.y[idx]/mesh.Ly)
            aμx[idx] = g
        end
    elseif objective == :kinetic_energy
        KE = zero(RT); Mtot = zero(RT)
        for ey in 1:N_ey, ex in 1:N_ex, j in 1:n_p, i in 1:n_p
            wJ = basis.w[i]*basis.w[j]*mesh.Jx*mesh.Jy
            KE   += wJ * (μxT[i,j,ex,ey]^2 + μyT[i,j,ex,ey]^2)/(2*ρT[i,j,ex,ey])
            Mtot += wJ * ρT[i,j,ex,ey]
        end
        for idx in eachindex(ρT)
            ux = μxT[idx]/ρT[idx]; uy = μyT[idx]/ρT[idx]
            aμx[idx] = ux / Mtot
            aμy[idx] = uy / Mtot
            aρ[idx]  = -(μxT[idx]^2 + μyT[idx]^2)/(2*ρT[idx]^2)/Mtot - KE/Mtot^2
        end
    elseif objective == :windowed_kinetic_energy
        # J = ∫ w(x,y) |μ|²/(2ρ) dxdy with the Gaussian window KE_WINDOW
        for idx in eachindex(ρT)
            w = ke_window(mesh.x[idx], mesh.y[idx])
            ux = μxT[idx]/ρT[idx]; uy = μyT[idx]/ρT[idx]
            aμx[idx] = w * ux
            aμy[idx] = w * uy
            aρ[idx]  = -w * (ux^2 + uy^2) / 2
        end
    else
        error("Unknown objective: $objective")
    end

    Π = zeros(RT, n_p, n_p, N_ex, N_ey)

    aμx1 = similar(aμx); aμy1 = similar(aμy); aρ1 = similar(aρ); aE1 = similar(aE)
    aμx2 = similar(aμx); aμy2 = similar(aμy); aρ2 = similar(aρ); aE2 = similar(aE)
    daμx = similar(aμx); daμy = similar(aμy); daρ = similar(aρ); daE = similar(aE)

    for n in n_steps:-1:1
        dt = min(Δt, T)
        μx_n, μy_n, ρ_n, E_n, Σ_n = snapshots[n+1]

        adj_rhs!(dx, dy, dr, de, ax, ay, ar, ae) =
            compute_adjoint_rhs_conservative!(dx, dy, dr, de, ax, ay, ar, ae,
                μx_n, μy_n, ρ_n, E_n, Σ_n, Π, basis, mesh, γ, α;
                n_iter=n_iter, solver=solver)

        adj_rhs!(daμx, daμy, daρ, daE, aμx, aμy, aρ, aE)
        @. aμx1 = aμx + dt*daμx; @. aμy1 = aμy + dt*daμy
        @. aρ1  = aρ  + dt*daρ;  @. aE1  = aE  + dt*daE

        adj_rhs!(daμx, daμy, daρ, daE, aμx1, aμy1, aρ1, aE1)
        @. aμx2 = 3/4*aμx + 1/4*aμx1 + 1/4*dt*daμx
        @. aμy2 = 3/4*aμy + 1/4*aμy1 + 1/4*dt*daμy
        @. aρ2  = 3/4*aρ  + 1/4*aρ1  + 1/4*dt*daρ
        @. aE2  = 3/4*aE  + 1/4*aE1  + 1/4*dt*daE

        adj_rhs!(daμx, daμy, daρ, daE, aμx2, aμy2, aρ2, aE2)
        @. aμx = 1/3*aμx + 2/3*aμx2 + 2/3*dt*daμx
        @. aμy = 1/3*aμy + 2/3*aμy2 + 2/3*dt*daμy
        @. aρ  = 1/3*aρ  + 2/3*aρ2  + 2/3*dt*daρ
        @. aE  = 1/3*aE  + 2/3*aE2  + 2/3*dt*daE
    end

    return aμx, aμy, aρ, aE
end
