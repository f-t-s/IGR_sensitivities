# ============================================================
# Initial conditions and error computation
# ============================================================

export init_entropy_wave, init_constant_state, l2_error,
       build_acoustic_pulse_ic!, build_entropy_wave_ic!, build_density_pulse_ic!,
       build_pressure_pulse_ic!, build_velocity_pulse_ic!,
       IC_BUILDERS, ic_dA_weights

"""
    init_entropy_wave(mesh, γ)

Entropy wave: a density perturbation advected at constant velocity.
  ρ(x,0) = 1 + 0.2 sin(2π x/L)
  u(x,0) = 1
  p(x,0) = 1

This is an exact solution to the Euler equations that advects
with velocity u=1. After time T=L (one period), the solution
returns to its initial state.

Returns (μ, ρ, E) as (p+1) × N_e arrays.
"""
function init_entropy_wave(mesh, γ)
    x = mesh.x
    L = mesh.L
    n_p, N_e = size(x)

    ρ = similar(x)
    μ = similar(x)
    E = similar(x)

    for e in 1:N_e
        for i in 1:n_p
            ρ[i,e] = 1.0 + 0.2 * sin(2π * x[i,e] / L)
            u = 1.0
            p = 1.0
            μ[i,e] = ρ[i,e] * u
            E[i,e] = p / (γ - 1) + 0.5 * ρ[i,e] * u^2
        end
    end

    return μ, ρ, E
end

"""
    init_constant_state(mesh, γ; ρ0=1.0, u0=1.0, p0=1.0)

Uniform constant state (for free-stream preservation test).
Returns (μ, ρ, E) as (p+1) × N_e arrays.
"""
function init_constant_state(mesh, γ; ρ0=1.0, u0=1.0, p0=1.0)
    x = mesh.x
    n_p, N_e = size(x)

    ρ = fill(ρ0, n_p, N_e)
    μ = fill(ρ0 * u0, n_p, N_e)
    E = fill(p0 / (γ - 1) + 0.5 * ρ0 * u0^2, n_p, N_e)

    return μ, ρ, E
end

"""
    l2_error(u, u_ref, basis, mesh)

Compute the L2 error ‖u - u_ref‖_L2 using LGL quadrature.
"""
function l2_error(u, u_ref, basis, mesh)
    N_e = mesh.N_e
    n_p = basis.p + 1
    J = mesh.J
    w = basis.w

    err2 = 0.0
    for e in 1:N_e
        for i in 1:n_p
            err2 += w[i] * J * (u[i,e] - u_ref[i,e])^2
        end
    end

    return sqrt(err2)
end

# ============================================================
# Parametric IC builders for adjoint/sensitivity tests
# ============================================================

"""
    build_acoustic_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)

Acoustic pulse: ρ = 1+A sin, u = A sin, p = 1+A sin.
All three primitive variables are perturbed in phase.
"""
function build_acoustic_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)
    n_p, N_e = size(mesh_x)
    for e in 1:N_e
        for i in 1:n_p
            x = mesh_x[i, e]
            s = sin(2π * x / L)
            ρ_val = 1 + A * s
            u_val = A * s
            p_val = 1 + A * s
            ρ0[i, e] = ρ_val
            μ0[i, e] = ρ_val * u_val
            E0[i, e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
    end
    return nothing
end

"""
    build_entropy_wave_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)

Entropy wave: ρ = 1+A sin, u = 1, p = 1.
Pure density perturbation advected at constant velocity.
Exact solution: ρ(x,t) = 1 + A sin(2π(x-t)/L).
"""
function build_entropy_wave_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)
    n_p, N_e = size(mesh_x)
    for e in 1:N_e
        for i in 1:n_p
            x = mesh_x[i, e]
            s = sin(2π * x / L)
            ρ_val = 1 + A * s
            u_val = 1.0
            p_val = 1.0
            ρ0[i, e] = ρ_val
            μ0[i, e] = ρ_val * u_val
            E0[i, e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
    end
    return nothing
end

"""
    build_density_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)

Density pulse: ρ = 1+A sin, u = 0, p = 1.
Only density is perturbed; zero velocity and uniform pressure.
Generates symmetric left/right-going acoustic waves.
"""
function build_density_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)
    n_p, N_e = size(mesh_x)
    for e in 1:N_e
        for i in 1:n_p
            x = mesh_x[i, e]
            s = sin(2π * x / L)
            ρ_val = 1 + A * s
            u_val = 0.0
            p_val = 1.0
            ρ0[i, e] = ρ_val
            μ0[i, e] = ρ_val * u_val
            E0[i, e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
    end
    return nothing
end

"""
    build_pressure_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)

Pressure pulse: ρ = 1, u = 0, p = 1+A sin.
Only pressure is perturbed; generates expanding acoustic waves.
"""
function build_pressure_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)
    n_p, N_e = size(mesh_x)
    for e in 1:N_e
        for i in 1:n_p
            x = mesh_x[i, e]
            s = sin(2π * x / L)
            ρ_val = 1.0
            u_val = 0.0
            p_val = 1 + A * s
            ρ0[i, e] = ρ_val
            μ0[i, e] = ρ_val * u_val
            E0[i, e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
    end
    return nothing
end

"""
    build_velocity_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)

Velocity pulse: ρ = 1, u = A sin, p = 1.
Only velocity is perturbed; uniform density and pressure.
"""
function build_velocity_pulse_ic!(μ0, ρ0, E0, A, mesh_x, L, γ)
    n_p, N_e = size(mesh_x)
    for e in 1:N_e
        for i in 1:n_p
            x = mesh_x[i, e]
            s = sin(2π * x / L)
            ρ_val = 1.0
            u_val = A * s
            p_val = 1.0
            ρ0[i, e] = ρ_val
            μ0[i, e] = ρ_val * u_val
            E0[i, e] = p_val / (γ - 1) + 0.5 * ρ_val * u_val^2
        end
    end
    return nothing
end

"""
    IC_BUILDERS

Dictionary mapping IC type symbols to (builder!, description) pairs.
Each builder has signature: builder!(μ0, ρ0, E0, A, mesh_x, L, γ).

Available IC types:
- `:acoustic_pulse` — ρ=1+A sin, u=A sin, p=1+A sin (coupled perturbation)
- `:entropy_wave`   — ρ=1+A sin, u=1, p=1 (advected density wave)
- `:density_pulse`  — ρ=1+A sin, u=0, p=1 (symmetric acoustic waves)
- `:pressure_pulse` — ρ=1, u=0, p=1+A sin (expanding pressure wave)
- `:velocity_pulse` — ρ=1, u=A sin, p=1 (velocity-only perturbation)
"""
const IC_BUILDERS = Dict{Symbol, Tuple{Function, String}}(
    :acoustic_pulse => (build_acoustic_pulse_ic!, "ρ=1+A sin, u=A sin, p=1+A sin"),
    :entropy_wave   => (build_entropy_wave_ic!,   "ρ=1+A sin, u=1, p=1"),
    :density_pulse  => (build_density_pulse_ic!,  "ρ=1+A sin, u=0, p=1"),
    :pressure_pulse => (build_pressure_pulse_ic!, "ρ=1, u=0, p=1+A sin"),
    :velocity_pulse => (build_velocity_pulse_ic!, "ρ=1, u=A sin, p=1"),
)

"""
    ic_dA_weights(ic_type::Symbol)

Return (wρ, wu, wp) indicating which primitive variables have ∂/∂A = sin(2πx/L).
Used to project adjoint fields onto dJ/dA = Σ (wρ·adj_ρ + wu·adj_u + wp·adj_p)·sin.
"""
function ic_dA_weights(ic_type::Symbol)
    if ic_type == :acoustic_pulse
        return (1.0, 1.0, 1.0)  # all three perturbed
    elseif ic_type == :entropy_wave
        return (1.0, 0.0, 0.0)  # only ρ perturbed
    elseif ic_type == :density_pulse
        return (1.0, 0.0, 0.0)  # only ρ perturbed
    elseif ic_type == :pressure_pulse
        return (0.0, 0.0, 1.0)  # only p perturbed
    elseif ic_type == :velocity_pulse
        return (0.0, 1.0, 0.0)  # only u perturbed
    else
        error("Unknown IC type: $ic_type")
    end
end
