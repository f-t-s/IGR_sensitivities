# ============================================================
# Strong-form tensor-product DG spatial operator for the
# 2D Euler equations on a Cartesian mesh.
# ============================================================

export compute_hyperbolic_rhs!

"""
    compute_hyperbolic_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ)

Compute the DG right-hand side for the hyperbolic part of the 2D Euler
equations using the strong form with LLF numerical fluxes.

All state arrays are `(n_p, n_p, N_ex, N_ey)`. Results are written into
the pre-allocated `dμx, dμy, dρ, dE`.

Strong form per element (separable tensor-product operator):
    dq = -(1/Jx) Dx·Fx - (1/Jy) Dy·Fy                       (volume)
    dq[n_p,j] -= (1/Jx)(f★ - Fx[n_p,j]) / w[n_p]   (x-right face)
    dq[1,j]   += (1/Jx)(f★ - Fx[1,j])   / w[1]     (x-left  face)
    dq[i,n_p] -= (1/Jy)(g★ - Fy[i,n_p]) / w[n_p]   (y-top   face)
    dq[i,1]   += (1/Jy)(g★ - Fy[i,1])   / w[1]     (y-bottom face)

`Dx` contracts the first node index `i`; `Dy` contracts the second `j`.
Both directions are periodic.
"""
function compute_hyperbolic_rhs!(dμx, dμy, dρ, dE, μx, μy, ρ, E, Σ, basis, mesh, γ)
    N_ex = mesh.N_ex
    N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    w    = basis.w
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy

    # Physical fluxes at all nodes (x- and y-directional)
    Fxμx = similar(μx); Fxμy = similar(μx); Fxρ = similar(μx); FxE = similar(μx)
    Fyμx = similar(μx); Fyμy = similar(μx); Fyρ = similar(μx); FyE = similar(μx)
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            mx = μx[i,j,ex,ey]; my = μy[i,j,ex,ey]
            r  = ρ[i,j,ex,ey];  en = E[i,j,ex,ey]; s = Σ[i,j,ex,ey]
            Fxμx[i,j,ex,ey], Fxμy[i,j,ex,ey], Fxρ[i,j,ex,ey], FxE[i,j,ex,ey] =
                polytropic_flux_x(γ, mx, my, r, en, s)
            Fyμx[i,j,ex,ey], Fyμy[i,j,ex,ey], Fyρ[i,j,ex,ey], FyE[i,j,ex,ey] =
                polytropic_flux_y(γ, mx, my, r, en, s)
        end
    end

    # Volume terms: dq = -(1/Jx) Dx·Fx - (1/Jy) Dy·Fy
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            aμx = zero(eltype(μx)); aμy = zero(eltype(μy))
            aρ  = zero(eltype(ρ));  aE  = zero(eltype(E))
            for k in 1:n_p
                # x-derivative: contract index i
                aμx -= invJx * D[i,k] * Fxμx[k,j,ex,ey]
                aμy -= invJx * D[i,k] * Fxμy[k,j,ex,ey]
                aρ  -= invJx * D[i,k] * Fxρ[k,j,ex,ey]
                aE  -= invJx * D[i,k] * FxE[k,j,ex,ey]
                # y-derivative: contract index j
                aμx -= invJy * D[j,k] * Fyμx[i,k,ex,ey]
                aμy -= invJy * D[j,k] * Fyμy[i,k,ex,ey]
                aρ  -= invJy * D[j,k] * Fyρ[i,k,ex,ey]
                aE  -= invJy * D[j,k] * FyE[i,k,ex,ey]
            end
            dμx[i,j,ex,ey] = aμx
            dμy[i,j,ex,ey] = aμy
            dρ[i,j,ex,ey]  = aρ
            dE[i,j,ex,ey]  = aE
        end
    end

    # x-normal faces: between element (ex,ey) and (ex+1,ey)
    for ey in 1:N_ey, ex in 1:N_ex
        exR = ex == N_ex ? 1 : ex + 1
        for j in 1:n_p
            fμx, fμy, fρ, fE = llf_flux_x(γ,
                μx[n_p,j,ex,ey], μy[n_p,j,ex,ey], ρ[n_p,j,ex,ey], E[n_p,j,ex,ey], Σ[n_p,j,ex,ey],
                μx[1,j,exR,ey],  μy[1,j,exR,ey],  ρ[1,j,exR,ey],  E[1,j,exR,ey],  Σ[1,j,exR,ey])

            # Correction on the left element's right face (node n_p)
            dμx[n_p,j,ex,ey] -= invJx * (fμx - Fxμx[n_p,j,ex,ey]) / w[n_p]
            dμy[n_p,j,ex,ey] -= invJx * (fμy - Fxμy[n_p,j,ex,ey]) / w[n_p]
            dρ[n_p,j,ex,ey]  -= invJx * (fρ  - Fxρ[n_p,j,ex,ey])  / w[n_p]
            dE[n_p,j,ex,ey]  -= invJx * (fE  - FxE[n_p,j,ex,ey])  / w[n_p]

            # Correction on the right element's left face (node 1)
            dμx[1,j,exR,ey] += invJx * (fμx - Fxμx[1,j,exR,ey]) / w[1]
            dμy[1,j,exR,ey] += invJx * (fμy - Fxμy[1,j,exR,ey]) / w[1]
            dρ[1,j,exR,ey]  += invJx * (fρ  - Fxρ[1,j,exR,ey])  / w[1]
            dE[1,j,exR,ey]  += invJx * (fE  - FxE[1,j,exR,ey])  / w[1]
        end
    end

    # y-normal faces: between element (ex,ey) and (ex,ey+1)
    for ey in 1:N_ey, ex in 1:N_ex
        eyR = ey == N_ey ? 1 : ey + 1
        for i in 1:n_p
            fμx, fμy, fρ, fE = llf_flux_y(γ,
                μx[i,n_p,ex,ey], μy[i,n_p,ex,ey], ρ[i,n_p,ex,ey], E[i,n_p,ex,ey], Σ[i,n_p,ex,ey],
                μx[i,1,ex,eyR],  μy[i,1,ex,eyR],  ρ[i,1,ex,eyR],  E[i,1,ex,eyR],  Σ[i,1,ex,eyR])

            # Correction on the bottom element's top face (node n_p)
            dμx[i,n_p,ex,ey] -= invJy * (fμx - Fyμx[i,n_p,ex,ey]) / w[n_p]
            dμy[i,n_p,ex,ey] -= invJy * (fμy - Fyμy[i,n_p,ex,ey]) / w[n_p]
            dρ[i,n_p,ex,ey]  -= invJy * (fρ  - Fyρ[i,n_p,ex,ey])  / w[n_p]
            dE[i,n_p,ex,ey]  -= invJy * (fE  - FyE[i,n_p,ex,ey])  / w[n_p]

            # Correction on the top element's bottom face (node 1)
            dμx[i,1,ex,eyR] += invJy * (fμx - Fyμx[i,1,ex,eyR]) / w[1]
            dμy[i,1,ex,eyR] += invJy * (fμy - Fyμy[i,1,ex,eyR]) / w[1]
            dρ[i,1,ex,eyR]  += invJy * (fρ  - Fyρ[i,1,ex,eyR])  / w[1]
            dE[i,1,ex,eyR]  += invJy * (fE  - FyE[i,1,ex,eyR])  / w[1]
        end
    end

    return nothing
end
