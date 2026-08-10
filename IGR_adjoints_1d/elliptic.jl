# ============================================================
# Matrix-free SIP (Symmetric Interior Penalty) elliptic solver
# for the IGR entropic pressure equation:
#   Σ/ρ - α ∂ₓ(∂ₓΣ/ρ) = R(q)
# ============================================================

export apply_sip!, compute_igr_rhs!, cg_solve!, solve_elliptic!,
       cg_fixed!, pcg_fixed!, jacobi_fixed!, chebyshev_fixed!, compute_sip_diagonal

"""
    apply_sip!(y, x, ρ, basis, mesh, α)

Apply the SIP bilinear form operator to vector `x`, writing result into `y`.
This computes y = A x where A corresponds to the weak form of:

  Σ/ρ - α ∂ₓ((1/ρ) ∂ₓΣ) = R

The bilinear form is:
  a(Σ, ψ) = Σ_e ∫_e (Σ/ρ)ψ dx  +  α Σ_e ∫_e (1/ρ)(∂ₓΣ)(∂ₓψ) dx
             - α Σ_f {{(1/ρ)∂Σ/∂n}}[[ψ]]
             - α Σ_f {{(1/ρ)∂ψ/∂n}}[[Σ]]
             + α Σ_f (σ_η/h_f) [[Σ]][[ψ]]

Convention at face between e_L (left) and e_R (right):
  n_L = +1 (outward normal of left element)
  n_R = -1 (outward normal of right element)
  [[v]] = v_L·n_L + v_R·n_R = v_L - v_R
  {{w}} = (w_L + w_R)/2

Arrays x, y, ρ are (p+1) × N_e.
"""
function apply_sip!(y, x, ρ, basis, mesh, α)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w
    invJ = 1 / J
    η = (basis.p + 1)^2  # penalty parameter

    fill!(y, zero(eltype(y)))

    # Volume contributions
    for e in 1:N_e
        for i in 1:n_p
            # Mass term: ∫(Σ/ρ)ψ dx ≈ Σ_i w[i]*J*(x[i]/ρ[i])*δ_{ik}
            y[i,e] += w[i] * J * x[i,e] / ρ[i,e]

            # Stiffness term: α ∫(1/ρ)(∂ₓΣ)(∂ₓψ) dx
            # ≈ α Σ_i w[i]*J * (1/ρ[i]) * (invJ*∂ξΣ[i]) * (invJ*D[i,k])
            # = α Σ_i w[i]*invJ * (1/ρ[i]) * ∂ξΣ[i] * D[i,k]
            dxi_x = zero(eltype(x))
            for j in 1:n_p
                dxi_x += D[i,j] * x[j,e]
            end
            for k in 1:n_p
                y[k,e] += α * w[i] * invJ * dxi_x / ρ[i,e] * D[i,k]
            end
        end
    end

    # Face contributions (SIP terms at each interior face)
    for e in 1:N_e
        e_R = e == N_e ? 1 : e + 1

        # Values at the interface
        Σ_L = x[n_p, e]
        Σ_R = x[1, e_R]
        ρ_L = ρ[n_p, e]
        ρ_R = ρ[1, e_R]

        # Jump [[Σ]] = Σ_L - Σ_R  (n_L=+1, n_R=-1)
        jump_Σ = Σ_L - Σ_R

        # Reference derivatives of Σ at face nodes
        dxi_Σ_L = zero(eltype(x))
        dxi_Σ_R = zero(eltype(x))
        for j in 1:n_p
            dxi_Σ_L += D[n_p, j] * x[j, e]
            dxi_Σ_R += D[1, j] * x[j, e_R]
        end

        # Average flux: {{(1/ρ)∂Σ/∂x}} · n_F  where n_F = +1 (fixed face normal)
        # = ((1/ρ_L)(invJ·dξΣ_L) + (1/ρ_R)(invJ·dξΣ_R)) / 2
        avg_flux = invJ * ((1/ρ_L) * dxi_Σ_L + (1/ρ_R) * dxi_Σ_R) / 2

        # Penalty coefficient: σ_η = η * max(κ_L, κ_R) / h_f
        κ_avg = max(1/ρ_L, 1/ρ_R)
        h_f = mesh.Δx
        σ_pen = η * κ_avg / h_f

        # --- Consistency: -α {{(1/ρ)∂Σ/∂x}}·n_F [[ψ]] ---
        # [[ψ]] at face: ψ_L·n_F - ψ_R·n_F with n_F=+1
        # For ψ at node n_p of e_L: [[ψ]] = +1
        y[n_p, e]  += -α * avg_flux * (+1)
        # For ψ at node 1 of e_R: [[ψ]] = -1
        y[1, e_R]  += -α * avg_flux * (-1)

        # --- Symmetry: -α {{(1/ρ)∂ψ/∂x}}·n_F [[Σ]] ---
        # {{(1/ρ)∂ψ_k/∂x}} for ψ_k in e_L: (1/(2ρ_L))*invJ*D[n_p,k]
        # {{(1/ρ)∂ψ_k/∂x}} for ψ_k in e_R: (1/(2ρ_R))*invJ*D[1,k]
        # n_F = +1, [[Σ]] = Σ_L - Σ_R
        for k in 1:n_p
            y[k, e]  += -α * (1/(2*ρ_L)) * invJ * D[n_p, k] * (+1) * jump_Σ
            y[k, e_R] += -α * (1/(2*ρ_R)) * invJ * D[1, k] * (+1) * jump_Σ
        end

        # --- Penalty: +α σ_η [[Σ]] [[ψ]] ---
        # For ψ at node n_p of e_L: [[ψ]] = +1
        y[n_p, e]  += α * σ_pen * jump_Σ
        # For ψ at node 1 of e_R: [[ψ]] = -1
        y[1, e_R]  -= α * σ_pen * jump_Σ
    end

    return nothing
end

"""
    compute_igr_rhs!(b, μ, ρ, E, basis, mesh, α)

Compute the right-hand side of the IGR elliptic equation:
  R(q) = 2α (∂u/∂x)² in 1D

projected onto the test space: b_i = ∫ R ψ_i dx (using LGL quadrature).
"""
function compute_igr_rhs!(b, μ, ρ, E, basis, mesh, α)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w
    invJ = 1 / J

    fill!(b, zero(eltype(b)))

    for e in 1:N_e
        # Compute velocity u = μ/ρ at each node
        # Then compute ∂u/∂x via D matrix
        for i in 1:n_p
            du_dx = zero(eltype(μ))
            for j in 1:n_p
                u_j = μ[j,e] / ρ[j,e]
                du_dx += D[i,j] * u_j
            end
            du_dx *= invJ  # physical derivative

            # R = 2α (∂u/∂x)²
            R_i = 2 * α * du_dx^2
            b[i,e] = w[i] * J * R_i
        end
    end

    return nothing
end

"""
    cg_solve!(x, apply_A!, b, tol, maxiter; work=nothing)

Conjugate gradient solver. Solves A x = b where `apply_A!(y, x)` computes y = A x.
`x` is the initial guess and is overwritten with the solution.
`b` is the right-hand side (not modified).
"""
function cg_solve!(x, apply_A!, b, tol, maxiter)
    r = similar(b)
    p = similar(b)
    Ap = similar(b)

    # r = b - A x
    apply_A!(Ap, x)
    @. r = b - Ap
    copyto!(p, r)
    rs_old = dot(r, r)

    for iter in 1:maxiter
        if sqrt(rs_old) < tol
            return iter
        end

        apply_A!(Ap, p)
        pAp = dot(p, Ap)
        α_cg = rs_old / pAp

        @. x = x + α_cg * p
        @. r = r - α_cg * Ap

        rs_new = dot(r, r)
        @. p = r + (rs_new / rs_old) * p

        rs_old = rs_new
    end

    return maxiter
end

"""
    compute_sip_diagonal(ρ, basis, mesh, α)

Compute the diagonal of the SIP operator analytically.
Used as the Jacobi preconditioner / iteration matrix.

The diagonal has three contributions per DOF (k, e):
1. Volume mass:      w[k] J / ρ[k,e]
2. Volume stiffness: α invJ Σ_i w[i] D[i,k]² / ρ[i,e]
3. Face penalty + consistency + symmetry (at boundary nodes only)
"""
function compute_sip_diagonal(ρ, basis, mesh, α)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w
    invJ = 1 / J
    η = n_p^2

    RT = promote_type(eltype(ρ), typeof(α))
    diag = zeros(RT, n_p, N_e)

    # Volume mass
    for e in 1:N_e, i in 1:n_p
        diag[i,e] += w[i] * J / ρ[i,e]
    end

    # Volume stiffness
    for e in 1:N_e, k in 1:n_p
        for i in 1:n_p
            diag[k,e] += α * w[i] * invJ * D[i,k]^2 / ρ[i,e]
        end
    end

    # Face contributions (SIP consistency + symmetry + penalty)
    for e in 1:N_e
        e_R = e == N_e ? 1 : e + 1
        ρ_L = ρ[n_p, e]
        ρ_R = ρ[1, e_R]

        κ_avg = max(1/ρ_L, 1/ρ_R)
        σ_pen = η * κ_avg / mesh.Δx

        # Left node of face (node n_p of element e)
        # Consistency + Symmetry (equal for SIP): -α invJ D[n_p,n_p] / ρ_L
        diag[n_p, e] += -α * invJ * D[n_p, n_p] / ρ_L
        # Penalty
        diag[n_p, e] += α * σ_pen

        # Right node of face (node 1 of element e_R)
        # Consistency + Symmetry: +α invJ D[1,1] / ρ_R
        diag[1, e_R] += α * invJ * D[1, 1] / ρ_R
        # Penalty
        diag[1, e_R] += α * σ_pen
    end

    return diag
end

"""
    cg_fixed!(x, apply_A!, b, n_iter)

Conjugate gradient solver with a **fixed** iteration count (no early exit).
Solves A x = b where `apply_A!(y, x)` computes y = A x.

Using a fixed number of iterations (instead of a convergence-based
stopping criterion) gives a well-defined, branch-free computation
graph, making this solver cleanly differentiable with ForwardDiff.
"""
function cg_fixed!(x, apply_A!, b, n_iter)
    r = similar(b)
    p = similar(b)
    Ap = similar(b)

    # r = b - A x
    apply_A!(Ap, x)
    @. r = b - Ap
    copyto!(p, r)
    rs_old = dot(r, r)

    for _ in 1:n_iter
        apply_A!(Ap, p)
        pAp = dot(p, Ap)
        # Guard against 0/0 when CG has converged (r ≈ 0, p ≈ 0)
        α_cg = rs_old / (pAp + 1e-30)

        @. x = x + α_cg * p
        @. r = r - α_cg * Ap

        rs_new = dot(r, r)
        β = rs_new / (rs_old + 1e-30)
        @. p = r + β * p

        rs_old = rs_new
    end

    return n_iter
end

"""
    pcg_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α)

Diagonally-preconditioned conjugate gradient with a **fixed** iteration count.
Solves A x = b where A is the SIP operator (applied via `apply_sip!`) and
`M_diag` is the diagonal of A (used as Jacobi preconditioner: M⁻¹r = r ./ M_diag).

The fixed iteration count keeps the computation graph branch-free,
making this solver cleanly differentiable with both ForwardDiff and Enzyme.
The SIP operator arguments are passed explicitly (no closures) for Enzyme
reverse-mode compatibility.
"""
function pcg_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α)
    r  = similar(b)
    z  = similar(b)
    p  = similar(b)
    Ap = similar(b)

    # r = b - A x
    apply_sip!(Ap, x, ρ, basis, mesh, α)
    @. r = b - Ap

    # z = M⁻¹ r
    @. z = r / M_diag
    copyto!(p, z)
    rz_old = dot(r, z)

    for _ in 1:n_iter
        apply_sip!(Ap, p, ρ, basis, mesh, α)
        pAp = dot(p, Ap)
        α_cg = rz_old / (pAp + 1e-30)

        @. x = x + α_cg * p
        @. r = r - α_cg * Ap
        @. z = r / M_diag

        rz_new = dot(r, z)
        β = rz_new / (rz_old + 1e-30)
        @. p = z + β * p

        rz_old = rz_new
    end

    return n_iter
end

"""
    jacobi_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α; ω=2/3)

Weighted Jacobi (Richardson) iteration with a **fixed** iteration count.
Solves A x = b where A is the SIP operator and M_diag is its diagonal.

Each iteration computes:
  x ← x + ω · (b - A x) ./ M_diag

Unlike PCG, this is a simple linear fixed-point iteration with no accumulated
state (no conjugacy directions or β updates). This makes it cleaner to
differentiate through with Enzyme reverse mode, avoiding the gradient
oscillations that PCG can exhibit at low iteration counts.

The relaxation parameter ω ∈ (0, 1] controls damping:
  ω = 1    — standard Jacobi (may diverge for stiff problems)
  ω = 2/3  — classical damped Jacobi (default, good for SIP)
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

Chebyshev semi-iterative method with Jacobi preconditioner and **fixed** iteration count.
Solves A x = b where A is the SIP operator and M_diag is its diagonal.

Uses 5 power iteration steps to estimate λ_max of M⁻¹A, then runs the
three-term Chebyshev recurrence with predetermined coefficients:
  d₁ = M⁻¹r / θ
  dₖ = ρₖ ρₖ₋₁ dₖ₋₁ + (2ρₖ/δ) M⁻¹rₖ     (k ≥ 2)
  xₖ = xₖ₋₁ + dₖ

where θ = (λ_max + λ_min)/2, δ = (λ_max - λ_min)/2, and ρₖ follows the
Chebyshev recurrence ρₖ = 1/(2σ - ρₖ₋₁) with σ = θ/δ.

Unlike PCG, the step sizes are predetermined (depend on eigenvalue estimates,
not on evolving inner products), giving a cleaner AD graph. Converges faster
than Jacobi iteration by a factor of √κ.
"""
function chebyshev_fixed!(x, b, M_diag, n_iter, ρ, basis, mesh, α)
    Ax = similar(b)
    r  = similar(b)
    z  = similar(b)
    d  = similar(b)

    # Eigenvalue bounds for Jacobi-preconditioned SIP operator M⁻¹A.
    # λ_min ≈ 1 (mass term dominates), λ_max saturates to a p-dependent
    # constant (penalty scaling cancels in preconditioning).
    n_p = size(x, 1)
    p = n_p - 1
    λ_min = 1.0
    λ_max = (1.5, 2.0, 2.1, 2.2)[min(p, 4)]

    θ = (λ_max + λ_min) / 2  # center
    δ = (λ_max - λ_min) / 2  # half-width
    σ = θ / max(δ, 1e-10)    # guard against δ ≈ 0

    # --- Chebyshev iteration ---
    apply_sip!(Ax, x, ρ, basis, mesh, α)
    @. r = b - Ax

    ρ_cheb = 1 / σ

    for k in 1:n_iter
        @. z = r / M_diag  # Jacobi preconditioner

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
    solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α; n_iter=10, solver=:pcg)

Solve the IGR elliptic equation for the entropic pressure Σ:
  Σ/ρ - α ∂ₓ(∂ₓΣ/ρ) = R(q)

Solver options:
  :pcg       — preconditioned CG (default, fast convergence)
  :jacobi    — damped Jacobi iteration (simplest AD graph)
  :chebyshev — Chebyshev-accelerated Jacobi (faster than Jacobi, cleaner AD than PCG)

All use a fixed iteration count (branch-free) for AD compatibility.
The input `Σ` is used as the initial guess (warm start).
"""
function solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α; n_iter=10, solver=:pcg)
    return solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α, n_iter, solver)
end

# Positional-argument form (avoids kwcall for Enzyme compatibility on Julia 1.12)
function solve_elliptic!(Σ, μ, ρ, E, basis, mesh, α, n_iter::Int, solver::Symbol=:pcg)
    b = similar(Σ)
    compute_igr_rhs!(b, μ, ρ, E, basis, mesh, α)

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
