# ============================================================
# 2D initial conditions and error computation
# ============================================================

export init_constant_state, init_isentropic_vortex, l2_error,
       build_taylor_green_ic!, build_sedov_blast_ic!, IC_BUILDERS, ic_dA_weights,
       build_oseen_pair_blast_ic!

"""
    init_constant_state(mesh, γ; ρ0=1.0, ux0=1.0, uy0=1.0, p0=1.0)

Uniform constant state, for the free-stream preservation test.
Returns (μx, μy, ρ, E) as (n_p, n_p, N_ex, N_ey) arrays.
"""
function init_constant_state(mesh, γ; ρ0=1.0, ux0=1.0, uy0=1.0, p0=1.0)
    sz = size(mesh.x)
    ρ  = fill(ρ0, sz)
    μx = fill(ρ0 * ux0, sz)
    μy = fill(ρ0 * uy0, sz)
    E  = fill(p0 / (γ - 1) + 0.5 * ρ0 * (ux0^2 + uy0^2), sz)
    return μx, μy, ρ, E
end

"""
    init_isentropic_vortex(mesh, γ; β=5.0, ux0=1.0, uy0=1.0)

Isentropic vortex — a smooth exact solution of the 2D Euler equations
that advects rigidly with the background velocity `(ux0, uy0)`.
On a periodic domain it returns to its initial state after one period.

The vortex is centered at the domain midpoint with strength `β`:
  δu  = -(y-y_c)(β/2π) exp((1-r²)/2)
  δv  = +(x-x_c)(β/2π) exp((1-r²)/2)
  δT  = -(γ-1)β²/(8γπ²) exp(1-r²)
  ρ   = (1+δT)^(1/(γ-1)),   p = ρ^γ

Returns (μx, μy, ρ, E).
"""
function init_isentropic_vortex(mesh, γ; β=5.0, ux0=1.0, uy0=1.0)
    x = mesh.x; y = mesh.y
    xc = mesh.Lx / 2
    yc = mesh.Ly / 2
    sz = size(x)
    μx = zeros(sz); μy = zeros(sz); ρ = zeros(sz); E = zeros(sz)

    for idx in eachindex(x)
        dx = x[idx] - xc
        dy = y[idx] - yc
        r2 = dx^2 + dy^2
        ef = exp((1 - r2) / 2)

        du = -dy * (β / (2π)) * ef
        dv =  dx * (β / (2π)) * ef
        dT = -(γ - 1) * β^2 / (8 * γ * π^2) * exp(1 - r2)

        ρ_val = (1 + dT)^(1 / (γ - 1))
        ux = ux0 + du
        uy = uy0 + dv
        p_val = ρ_val^γ

        ρ[idx]  = ρ_val
        μx[idx] = ρ_val * ux
        μy[idx] = ρ_val * uy
        E[idx]  = p_val / (γ - 1) + 0.5 * ρ_val * (ux^2 + uy^2)
    end
    return μx, μy, ρ, E
end

"""
    l2_error(u, u_ref, basis, mesh)

L² error ‖u - u_ref‖ using the tensor-product LGL quadrature rule.
"""
function l2_error(u, u_ref, basis, mesh)
    N_ex = mesh.N_ex; N_ey = mesh.N_ey
    n_p  = basis.p + 1
    w    = basis.w
    JxJy = mesh.Jx * mesh.Jy

    err2 = zero(eltype(u))
    for ey in 1:N_ey, ex in 1:N_ex
        for j in 1:n_p, i in 1:n_p
            err2 += w[i] * w[j] * JxJy * (u[i,j,ex,ey] - u_ref[i,j,ex,ey])^2
        end
    end
    return sqrt(err2)
end

# ============================================================
# Parametric IC builders for adjoint / sensitivity tests
# ============================================================

"""
    build_taylor_green_ic!(μx0, μy0, ρ0, E0, A, mesh, γ)

Taylor-Green-style velocity field of amplitude `A`:
  ux =  A sin(2πx/Lx) cos(2πy/Ly)
  uy = -A cos(2πx/Lx) sin(2πy/Ly)
  ρ  = 1,  p = 1.

The velocity field has nonzero gradients, so the IGR source `R(q)` and
hence `Σ` are nonzero — making this a useful parametric IC for the
adjoint and sensitivity comparisons (parameter = amplitude `A`).
"""
function build_taylor_green_ic!(μx0, μy0, ρ0, E0, A, mesh, γ)
    x = mesh.x; y = mesh.y
    Lx = mesh.Lx; Ly = mesh.Ly
    for idx in eachindex(x)
        cx = 2π * x[idx] / Lx
        cy = 2π * y[idx] / Ly
        ux =  A * sin(cx) * cos(cy)
        uy = -A * cos(cx) * sin(cy)
        ρ_val = 1.0
        p_val = 1.0
        ρ0[idx]  = ρ_val
        μx0[idx] = ρ_val * ux
        μy0[idx] = ρ_val * uy
        E0[idx]  = p_val / (γ - 1) + 0.5 * ρ_val * (ux^2 + uy^2)
    end
    return nothing
end

"""
    build_sedov_blast_ic!(μx0, μy0, ρ0, E0, A, mesh, γ; p_bg=1.0, σ=0.1)

Smoothed Sedov-Taylor blast wave: a centered Gaussian over-pressure of
amplitude `A` in an ambient gas at rest.
  ρ  = 1,  u = 0
  p  = p_bg + A·exp(-r²/σ²),   r² = (x-x_c)² + (y-y_c)²

The over-pressured core drives a radially-expanding blast wave; the IGR
regularization keeps the steepening front smooth. Only the energy (via
the pressure) depends on the amplitude `A`, so `A` is a valid IC
parameter for the adjoint / sensitivity comparisons.
"""
function build_sedov_blast_ic!(μx0, μy0, ρ0, E0, A, mesh, γ; p_bg=1.0, σ=0.1)
    xc = mesh.Lx / 2
    yc = mesh.Ly / 2
    for idx in eachindex(mesh.x)
        dx = mesh.x[idx] - xc
        dy = mesh.y[idx] - yc
        r2 = dx^2 + dy^2
        p_val = p_bg + A * exp(-r2 / σ^2)
        ρ0[idx]  = 1.0
        μx0[idx] = 0.0
        μy0[idx] = 0.0
        E0[idx]  = p_val / (γ - 1)
    end
    return nothing
end

"""
    build_oseen_pair_blast_ic!(μx0, μy0, ρ0, E0, A, mesh, γ;
                               a_core=0.05, M_star=0.5,
                               vort_centers=((0.45, 0.60, +1), (0.45, 0.40, -1)),
                               blast_xc=0.25, blast_yc=0.50, σ=0.09)

Counter-rotating pair of "physical" Oseen vortices (Colonius, Lele &
Moin, JFM 1991, §3.1 initialization) plus a Gaussian over-pressure
blast of amplitude `A`.

Each vortex has Gaussian vorticity, tangential velocity
    v_θ(r) = Γ/(2πr)·(1 - exp(-r²/a²)),
and homentropic radial-equilibrium thermodynamics obtained by
integrating dh/dr = v_θ²/r for the enthalpy h = γ/(γ-1)·p^((γ-1)/γ)
(tabulated enthalpy defect, composed additively for the pair). The
circulation Γ is set from the vortex Mach number `M_star` = v_max/a∞
via v_max = 0.6383·Γ/(2πa). Velocity is summed over 3×3 periodic
images; the pair carries zero net circulation as required by the
doubly-periodic domain. `vort_centers` holds (xc, yc, sign) triples.

Only the energy depends on `A` (∂E0/∂A = exp(-r_b²/σ²)/(γ-1)), so `A`
is a valid IC parameter for adjoint / FD sensitivity comparisons.
"""
function build_oseen_pair_blast_ic!(μx0, μy0, ρ0, E0, A, mesh, γ;
                                    a_core=0.05, M_star=0.5,
                                    vort_centers=((0.45, 0.60, +1), (0.45, 0.40, -1)),
                                    blast_xc=0.25, blast_yc=0.50, σ=0.09)
    L = mesh.Lx
    Γ_circ = M_star * sqrt(γ) * 2π * a_core / 0.6383

    # enthalpy-defect table  D(r) = (Γ/2π)² ∫_r^∞ (1-e^{-s²/a²})²/s³ ds
    r_max = 2.0 * L
    n_tab = 4000
    r_tab = range(0, r_max, length=n_tab)
    D_tab = zeros(n_tab)
    pref = (Γ_circ / (2π))^2
    acc = 0.0
    for i in n_tab-1:-1:1
        s1, s2 = r_tab[i], r_tab[i+1]
        f1 = s1 < 1e-12 ? 0.0 : (1.0 - exp(-s1^2 / a_core^2))^2 / s1^3
        f2 = (1.0 - exp(-s2^2 / a_core^2))^2 / s2^3
        acc += 0.5 * (f1 + f2) * (s2 - s1)
        D_tab[i] = pref * acc
    end
    defect(r) = begin
        r >= r_max && return 0.0
        t = r / r_max * (n_tab - 1) + 1
        i = clamp(floor(Int, t), 1, n_tab - 1)
        w = t - i
        (1 - w) * D_tab[i] + w * D_tab[i+1]
    end

    h_inf = γ / (γ - 1)
    for idx in eachindex(mesh.x)
        x = mesh.x[idx]; y = mesh.y[idx]
        ux = 0.0; uy = 0.0; h = h_inf
        for (xc, yc, sgn) in vort_centers
            Γk = sgn * Γ_circ
            for sx in -1:1, sy in -1:1
                dx = x - (xc + sx * L); dy = y - (yc + sy * L)
                r2 = dx^2 + dy^2
                r2 < 1e-24 && continue
                vθ_over_r = Γk / (2π * r2) * (1.0 - exp(-r2 / a_core^2))
                ux += -vθ_over_r * dy
                uy +=  vθ_over_r * dx
            end
            dx = x - xc; dy = y - yc
            dx -= L * round(dx / L); dy -= L * round(dy / L)
            h -= defect(sqrt(dx^2 + dy^2))
        end
        p_vort = ((γ - 1) * h / γ)^(γ / (γ - 1))
        ρ_val  = p_vort^(1.0 / γ)
        r2b    = (x - blast_xc)^2 + (y - blast_yc)^2
        p_val  = p_vort + A * exp(-r2b / σ^2)
        ρ0[idx]  = ρ_val
        μx0[idx] = ρ_val * ux
        μy0[idx] = ρ_val * uy
        E0[idx]  = p_val / (γ - 1) + 0.5 * ρ_val * (ux^2 + uy^2)
    end
    return nothing
end

"""
    IC_BUILDERS

Dictionary mapping IC-type symbols to (builder!, description) pairs.
Each builder has signature `builder!(μx0, μy0, ρ0, E0, A, mesh, γ)`.
"""
const IC_BUILDERS = Dict{Symbol, Tuple{Function, String}}(
    :taylor_green => (build_taylor_green_ic!, "ux=A sin cos, uy=-A cos sin, ρ=1, p=1"),
    :sedov_blast  => (build_sedov_blast_ic!,  "ρ=1, u=0, p=p_bg+A exp(-r²/σ²)"),
)

"""
    ic_dA_weights(ic_type::Symbol)

Return `(wρ, wux, wuy, wp)` flags marking which primitive variables carry
the ∂/∂A perturbation. Used to project the adjoint fields onto dJ/dA.
For `:taylor_green` only the velocity components depend on `A`.
"""
function ic_dA_weights(ic_type::Symbol)
    if ic_type == :taylor_green
        return (0.0, 1.0, 1.0, 0.0)
    elseif ic_type == :sedov_blast
        return (0.0, 0.0, 0.0, 1.0)   # only the pressure depends on A
    else
        error("Unknown IC type: $ic_type")
    end
end
