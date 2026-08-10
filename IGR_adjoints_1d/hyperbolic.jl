# ============================================================
# Strong-form DG spatial operator for the 1D Euler equations
# ============================================================

export compute_hyperbolic_rhs!

"""
    compute_hyperbolic_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ)

Compute the DG right-hand side for the hyperbolic part of the Euler
equations using the strong form with LLF numerical flux.

All arrays are (p+1) × N_e. Writes into pre-allocated dμ, dρ, dE.

Strong form per element e:
    dq[:,e] = -(1/J) D f[:,e]                                 (volume)
    dq[end,e] -= (1/J)(f★_right - f[end,e]) / w[end]         (right face)
    dq[1,e]   += (1/J)(f★_left  - f[1,e])   / w[1]           (left face)

Equivalent to the weak form via the SBP property Dᵀ W + W D = B,
which holds exactly when quadrature and interpolation nodes coincide.
"""
function compute_hyperbolic_rhs!(dμ, dρ, dE, μ, ρ, E, Σ, basis, mesh, γ)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    D = basis.D
    w = basis.w

    # Compute physical flux at all nodes
    fμ = similar(μ)
    fρ = similar(ρ)
    fE = similar(E)
    for e in 1:N_e
        for i in 1:n_p
            fμ[i,e], fρ[i,e], fE[i,e] = polytropic_flux(γ, μ[i,e], ρ[i,e], E[i,e], Σ[i,e])
        end
    end

    # Volume terms: dq = -(1/J) D f
    invJ = 1 / J
    for e in 1:N_e
        for i in 1:n_p
            dμ[i,e] = zero(eltype(μ))
            dρ[i,e] = zero(eltype(ρ))
            dE[i,e] = zero(eltype(E))
            for j in 1:n_p
                dμ[i,e] -= invJ * D[i,j] * fμ[j,e]
                dρ[i,e] -= invJ * D[i,j] * fρ[j,e]
                dE[i,e] -= invJ * D[i,j] * fE[j,e]
            end
        end
    end

    # Face fluxes and corrections
    for e in 1:N_e
        # Periodic neighbor indices
        e_left  = e == 1 ? N_e : e - 1
        e_right = e == N_e ? 1 : e + 1

        # Right face of element e: interface between e (right) and e_right (left)
        fstar_μR, fstar_ρR, fstar_ER = llf_flux(γ,
            μ[n_p,e], ρ[n_p,e], E[n_p,e], Σ[n_p,e],
            μ[1,e_right], ρ[1,e_right], E[1,e_right], Σ[1,e_right])

        # Left face of element e: interface between e_left (right) and e (left)
        fstar_μL, fstar_ρL, fstar_EL = llf_flux(γ,
            μ[n_p,e_left], ρ[n_p,e_left], E[n_p,e_left], Σ[n_p,e_left],
            μ[1,e], ρ[1,e], E[1,e], Σ[1,e])

        # Right face correction: subtract (f★ - f_internal) / w at last node
        dμ[n_p,e] -= invJ * (fstar_μR - fμ[n_p,e]) / w[n_p]
        dρ[n_p,e] -= invJ * (fstar_ρR - fρ[n_p,e]) / w[n_p]
        dE[n_p,e] -= invJ * (fstar_ER - fE[n_p,e]) / w[n_p]

        # Left face correction: add (f★ - f_internal) / w at first node
        dμ[1,e] += invJ * (fstar_μL - fμ[1,e]) / w[1]
        dρ[1,e] += invJ * (fstar_ρL - fρ[1,e]) / w[1]
        dE[1,e] += invJ * (fstar_EL - fE[1,e]) / w[1]
    end

    return nothing
end
