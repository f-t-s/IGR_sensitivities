# ============================================================
# Matrix-free 2D SIP (Symmetric Interior Penalty) elliptic solver
# for the IGR entropic pressure equation:
#   Σ/ρ - α ∇·(∇Σ/ρ) = R(q)
# on a doubly-periodic Cartesian mesh with tensor-product DG.
# ============================================================

export apply_sip!, compute_igr_rhs!, compute_sip_diagonal,
       cg_solve!, cg_fixed!, pcg_fixed!, jacobi_fixed!, chebyshev_fixed!,
       solve_elliptic!

"""
    apply_sip!(y, x, ρ, basis, mesh, α)

Apply the 2D SIP bilinear-form operator: `y = A x`, where `A` is the
weak form of `Σ/ρ - α ∇·(∇Σ/ρ)`.

The bilinear form, summed over elements `K` and faces `f`, is
  a(Σ,ψ) = Σ_K ∫_K (Σ/ρ)ψ
         + α Σ_K ∫_K (∇Σ·∇ψ)/ρ
         - α Σ_f ∫_f {{(∇Σ/ρ)·n}} [[ψ]]
         - α Σ_f ∫_f {{(∇ψ/ρ)·n}} [[Σ]]
         + α Σ_f ∫_f (η/h_f){{1/ρ}} [[Σ]][[ψ]].

Volume integrals use the tensor-product LGL rule (weight w[i]w[j]·Jx·Jy);
face integrals use the 1D edge rule (weight w·J along the edge).

Arrays `x, y, ρ` are `(n_p, n_p, N_ex, N_ey)`.
"""
function apply_sip!(y, x, ρ, basis, mesh, α)
    N_ex = mesh.N_ex
    N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    w    = basis.w
    Jx   = mesh.Jx
    Jy   = mesh.Jy
    invJx = 1 / Jx
    invJy = 1 / Jy
    η = (basis.p + 1)^2  # penalty parameter

    fill!(y, zero(eltype(y)))

    # --- Volume contributions: mass + stiffness ---
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            # Mass term: ∫(Σ/ρ)ψ  ≈  w[i]w[j]·Jx·Jy · x/ρ  (diagonal)
            y[i,j,ex,ey] += w[i] * w[j] * Jx * Jy * x[i,j,ex,ey] / ρ[i,j,ex,ey]

            # Reference derivatives ∂ξx, ∂ηx at node (i,j)
            dξx = zero(eltype(x))
            dηx = zero(eltype(x))
            for k in 1:n_p
                dξx += D[i,k] * x[k,j,ex,ey]
                dηx += D[j,k] * x[i,k,ex,ey]
            end
            invρ = 1 / ρ[i,j,ex,ey]

            # x-stiffness: α ∫ (1/ρ)(∂xΣ)(∂xψ); ∂x = invJx·∂ξ.
            # Factor w[i]w[j]·Jx·Jy·invJx² = w[i]w[j]·Jy/Jx.
            cx = α * w[i] * w[j] * Jy * invJx * invρ * dξx
            # y-stiffness: factor w[i]w[j]·Jx/Jy.
            cy = α * w[i] * w[j] * Jx * invJy * invρ * dηx
            for k in 1:n_p
                y[k,j,ex,ey] += cx * D[i,k]
                y[i,k,ex,ey] += cy * D[j,k]
            end
        end
    end

    # --- x-normal interior faces (edge indexed by j) ---
    for ey in 1:N_ey, ex in 1:N_ex
        exR = ex == N_ex ? 1 : ex + 1
        for j in 1:n_p
            wf = w[j] * Jy  # edge quadrature weight

            Σ_L = x[n_p,j,ex,ey]
            Σ_R = x[1,j,exR,ey]
            ρ_L = ρ[n_p,j,ex,ey]
            ρ_R = ρ[1,j,exR,ey]
            jump_Σ = Σ_L - Σ_R

            dξΣ_L = zero(eltype(x))
            dξΣ_R = zero(eltype(x))
            for k in 1:n_p
                dξΣ_L += D[n_p,k] * x[k,j,ex,ey]
                dξΣ_R += D[1,k]   * x[k,j,exR,ey]
            end

            avg_flux = invJx * ((1/ρ_L) * dξΣ_L + (1/ρ_R) * dξΣ_R) / 2
            σ_pen = η * max(1/ρ_L, 1/ρ_R) / mesh.Δx

            # Consistency: -α {{(∇Σ/ρ)·n}} [[ψ]]
            y[n_p,j,ex,ey] += -α * avg_flux * wf
            y[1,j,exR,ey]  +=  α * avg_flux * wf

            # Symmetry: -α {{(∇ψ/ρ)·n}} [[Σ]]
            for k in 1:n_p
                y[k,j,ex,ey]  += -α * (1/(2*ρ_L)) * invJx * D[n_p,k] * jump_Σ * wf
                y[k,j,exR,ey] += -α * (1/(2*ρ_R)) * invJx * D[1,k]   * jump_Σ * wf
            end

            # Penalty: +α σ [[Σ]][[ψ]]
            y[n_p,j,ex,ey] += α * σ_pen * jump_Σ * wf
            y[1,j,exR,ey]  -= α * σ_pen * jump_Σ * wf
        end
    end

    # --- y-normal interior faces (edge indexed by i) ---
    for ey in 1:N_ey, ex in 1:N_ex
        eyR = ey == N_ey ? 1 : ey + 1
        for i in 1:n_p
            wf = w[i] * Jx

            Σ_L = x[i,n_p,ex,ey]
            Σ_R = x[i,1,ex,eyR]
            ρ_L = ρ[i,n_p,ex,ey]
            ρ_R = ρ[i,1,ex,eyR]
            jump_Σ = Σ_L - Σ_R

            dηΣ_L = zero(eltype(x))
            dηΣ_R = zero(eltype(x))
            for k in 1:n_p
                dηΣ_L += D[n_p,k] * x[i,k,ex,ey]
                dηΣ_R += D[1,k]   * x[i,k,ex,eyR]
            end

            avg_flux = invJy * ((1/ρ_L) * dηΣ_L + (1/ρ_R) * dηΣ_R) / 2
            σ_pen = η * max(1/ρ_L, 1/ρ_R) / mesh.Δy

            y[i,n_p,ex,ey] += -α * avg_flux * wf
            y[i,1,ex,eyR]  +=  α * avg_flux * wf

            for k in 1:n_p
                y[i,k,ex,ey]  += -α * (1/(2*ρ_L)) * invJy * D[n_p,k] * jump_Σ * wf
                y[i,k,ex,eyR] += -α * (1/(2*ρ_R)) * invJy * D[1,k]   * jump_Σ * wf
            end

            y[i,n_p,ex,ey] += α * σ_pen * jump_Σ * wf
            y[i,1,ex,eyR]  -= α * σ_pen * jump_Σ * wf
        end
    end

    return nothing
end

"""
    compute_igr_rhs!(b, μx, μy, ρ, E, basis, mesh, α)

Assemble the discrete RHS of the IGR elliptic equation,
`b_ij = ∫ R(q) ψ_ij`, with the general-dimension IGR source
(adjoints.tex eq. igr_euler, d=2):

  R = α [ (∇·u)² + tr((Du)²) ]
    = α [ (uₓₓ + uᵧᵧ)² + uₓₓ² + 2 uₓᵧ uᵧₓ + uᵧᵧ² ]

where uₓₓ=∂ux/∂x, uᵧᵧ=∂uy/∂y, uₓᵧ=∂ux/∂y, uᵧₓ=∂uy/∂x.
"""
function compute_igr_rhs!(b, μx, μy, ρ, E, basis, mesh, α)
    N_ex = mesh.N_ex
    N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    w    = basis.w
    invJx = 1 / mesh.Jx
    invJy = 1 / mesh.Jy

    fill!(b, zero(eltype(b)))

    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            uxx = zero(eltype(μx))  # ∂ux/∂x
            uxy = zero(eltype(μx))  # ∂ux/∂y
            uyx = zero(eltype(μx))  # ∂uy/∂x
            uyy = zero(eltype(μx))  # ∂uy/∂y
            for k in 1:n_p
                ux_kj = μx[k,j,ex,ey] / ρ[k,j,ex,ey]
                uy_kj = μy[k,j,ex,ey] / ρ[k,j,ex,ey]
                ux_ik = μx[i,k,ex,ey] / ρ[i,k,ex,ey]
                uy_ik = μy[i,k,ex,ey] / ρ[i,k,ex,ey]
                uxx += D[i,k] * ux_kj
                uyx += D[i,k] * uy_kj
                uxy += D[j,k] * ux_ik
                uyy += D[j,k] * uy_ik
            end
            uxx *= invJx
            uyx *= invJx
            uxy *= invJy
            uyy *= invJy

            R = α * ((uxx + uyy)^2 + uxx^2 + 2 * uxy * uyx + uyy^2)
            b[i,j,ex,ey] = w[i] * w[j] * mesh.Jx * mesh.Jy * R
        end
    end

    return nothing
end

"""
    compute_sip_diagonal(ρ, basis, mesh, α)

Diagonal of the 2D SIP operator, used as Jacobi preconditioner /
iteration matrix. Mirrors `apply_sip!` but keeps only self-coupling.
"""
function compute_sip_diagonal(ρ, basis, mesh, α)
    N_ex = mesh.N_ex
    N_ey = mesh.N_ey
    n_p  = basis.p + 1
    D    = basis.D
    w    = basis.w
    Jx   = mesh.Jx
    Jy   = mesh.Jy
    invJx = 1 / Jx
    invJy = 1 / Jy
    η = n_p^2

    RT = promote_type(eltype(ρ), typeof(α))
    diag = zeros(RT, n_p, n_p, N_ex, N_ey)

    # Volume mass + stiffness
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            diag[i,j,ex,ey] += w[i] * w[j] * Jx * Jy / ρ[i,j,ex,ey]
        end
        for l in 1:n_p, k in 1:n_p
            # x-stiffness self-coupling: Σ_i w[i]w[l](Jy/Jx) D[i,k]²/ρ[i,l]
            for i in 1:n_p
                diag[k,l,ex,ey] += α * w[i] * w[l] * Jy * invJx * D[i,k]^2 / ρ[i,l,ex,ey]
            end
            # y-stiffness self-coupling: Σ_j w[k]w[j](Jx/Jy) D[j,l]²/ρ[k,j]
            for j in 1:n_p
                diag[k,l,ex,ey] += α * w[k] * w[j] * Jx * invJy * D[j,l]^2 / ρ[k,j,ex,ey]
            end
        end
    end

    # x-face contributions (consistency+symmetry+penalty at edge nodes)
    for ey in 1:N_ey, ex in 1:N_ex
        exR = ex == N_ex ? 1 : ex + 1
        for j in 1:n_p
            wf = w[j] * Jy
            ρ_L = ρ[n_p,j,ex,ey]
            ρ_R = ρ[1,j,exR,ey]
            σ_pen = η * max(1/ρ_L, 1/ρ_R) / mesh.Δx
            diag[n_p,j,ex,ey] += (-α * invJx * D[n_p,n_p] / ρ_L + α * σ_pen) * wf
            diag[1,j,exR,ey]  += ( α * invJx * D[1,1]     / ρ_R + α * σ_pen) * wf
        end
    end

    # y-face contributions
    for ey in 1:N_ey, ex in 1:N_ex
        eyR = ey == N_ey ? 1 : ey + 1
        for i in 1:n_p
            wf = w[i] * Jx
            ρ_L = ρ[i,n_p,ex,ey]
            ρ_R = ρ[i,1,ex,eyR]
            σ_pen = η * max(1/ρ_L, 1/ρ_R) / mesh.Δy
            diag[i,n_p,ex,ey] += (-α * invJy * D[n_p,n_p] / ρ_L + α * σ_pen) * wf
            diag[i,1,ex,eyR]  += ( α * invJy * D[1,1]     / ρ_R + α * σ_pen) * wf
        end
    end

    return diag
end

# ============================================================
# Iterative solvers — dimension-agnostic (operate on the
# flattened arrays via `apply_sip!` and `dot`). Fixed-iteration
# variants are branch-free for clean AD graphs.
# ============================================================

"""
    cg_solve!(x, apply_A!, b, tol, maxiter)

Conjugate gradient with convergence-based stopping. `apply_A!(y,x)`
computes `y = A x`. `x` is the initial guess, overwritten with the solution.
"""
function cg_solve!(x, apply_A!, b, tol, maxiter)
    r = similar(b); p = similar(b); Ap = similar(b)
    apply_A!(Ap, x)
    @. r = b - Ap
    copyto!(p, r)
    rs_old = dot(r, r)
    for iter in 1:maxiter
        if sqrt(rs_old) < tol
            return iter
        end
        apply_A!(Ap, p)
        α_cg = rs_old / dot(p, Ap)
        @. x = x + α_cg * p
        @. r = r - α_cg * Ap
        rs_new = dot(r, r)
        @. p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    end
    return maxiter
end

"""
    cg_fixed!(x, apply_A!, b, n_iter)

Conjugate gradient with a fixed iteration count (branch-free for AD).
"""
function cg_fixed!(x, apply_A!, b, n_iter)
    r = similar(b); p = similar(b); Ap = similar(b)
    apply_A!(Ap, x)
    @. r = b - Ap
    copyto!(p, r)
    rs_old = dot(r, r)
    for _ in 1:n_iter
        apply_A!(Ap, p)
        α_cg = rs_old / (dot(p, Ap) + 1e-30)
        @. x = x + α_cg * p
        @. r = r - α_cg * Ap
        rs_new = dot(r, r)
        @. p = r + (rs_new / (rs_old + 1e-30)) * p
        rs_old = rs_new
    end
    return n_iter
end

"""
    pcg_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α)

Diagonally-preconditioned CG with a fixed iteration count. The SIP
operator arguments are passed explicitly (no closures) for Enzyme
reverse-mode compatibility.
"""
function pcg_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α)
    r  = similar(b); z = similar(b); p = similar(b); Ap = similar(b)
    apply_sip!(Ap, x, ρ, basis, mesh, α)
    @. r = b - Ap
    @. z = r / M_diag
    copyto!(p, z)
    rz_old = dot(r, z)
    for _ in 1:n_iter
        apply_sip!(Ap, p, ρ, basis, mesh, α)
        α_cg = rz_old / (dot(p, Ap) + 1e-30)
        @. x = x + α_cg * p
        @. r = r - α_cg * Ap
        @. z = r / M_diag
        rz_new = dot(r, z)
        @. p = z + (rz_new / (rz_old + 1e-30)) * p
        rz_old = rz_new
    end
    return n_iter
end

"""
    jacobi_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α, ω)

Weighted Jacobi (Richardson) iteration with a fixed iteration count.
"""
function jacobi_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α, ω)
    Ax = similar(b)
    for _ in 1:n_iter
        apply_sip!(Ax, x, ρ, basis, mesh, α)
        @. x = x + ω * (b - Ax) / M_diag
    end
    return n_iter
end

"""
    chebyshev_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α)

Chebyshev semi-iterative method with Jacobi preconditioner and a fixed
iteration count. Step sizes are predetermined from eigenvalue bounds of
the preconditioned operator M⁻¹A, giving a clean AD graph.
"""
function chebyshev_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α)
    Ax = similar(b); r = similar(b); z = similar(b); d = similar(b)

    p = size(x, 1) - 1
    λ_min = 1.0
    λ_max = (1.5, 2.0, 2.1, 2.2)[min(p, 4)]
    θ = (λ_max + λ_min) / 2
    δ = (λ_max - λ_min) / 2
    σ = θ / max(δ, 1e-10)

    apply_sip!(Ax, x, ρ, basis, mesh, α)
    @. r = b - Ax
    ρ_cheb = 1 / σ
    for k in 1:n_iter
        @. z = r / M_diag
        if k == 1
            @. d = z / θ
        else
            ρ_new = 1 / (2σ - ρ_cheb)
            @. d = ρ_new * ρ_cheb * d + (2 * ρ_new / δ) * z
            ρ_cheb = ρ_new
        end
        @. x = x + d
        apply_sip!(Ax, x, ρ, basis, mesh, α)
        @. r = b - Ax
    end
    return n_iter
end

"""
    solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α; n_iter=10, solver=:pcg)

Solve the IGR elliptic equation for the entropic pressure Σ:
  Σ/ρ - α ∇·(∇Σ/ρ) = R(q).

Solver options: `:pcg`, `:jacobi`, `:chebyshev` — all fixed-iteration.
`Σ` is used as the initial guess (warm start).
"""
function solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α; n_iter=10, solver=:pcg)
    return solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α, n_iter, solver)
end

# Positional-argument form (avoids kwcall for Enzyme compatibility)
function solve_elliptic!(Σ, μx, μy, ρ, E, basis, mesh, α, n_iter::Int, solver::Symbol=:pcg)
    b = similar(Σ)
    compute_igr_rhs!(b, μx, μy, ρ, E, basis, mesh, α)
    M_diag = compute_sip_diagonal(ρ, basis, mesh, α)
    if solver == :pcg
        pcg_fixed!(Σ, b, M_diag, n_iter, ρ, basis, mesh, α)
    elseif solver == :jacobi
        jacobi_fixed!(Σ, b, M_diag, n_iter, ρ, basis, mesh, α, 2.0/3.0)
    elseif solver == :chebyshev
        chebyshev_fixed!(Σ, b, M_diag, n_iter, ρ, basis, mesh, α)
    else
        error("Unknown solver: $solver (use :pcg, :jacobi, or :chebyshev)")
    end
    return n_iter
end
