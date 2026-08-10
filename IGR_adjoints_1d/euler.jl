# ============================================================
# 1D Euler equations: flux, EOS, sound speed, numerical flux
# All functions are type-generic (no ::Float64 annotations)
# ============================================================

export polytropic_pressure, polytropic_flux, sound_speed, max_wavespeed, llf_flux

"""
    polytropic_pressure(γ, μ, ρ, E)

Compute the polytropic (ideal gas) pressure: p = (γ-1)(E - μ²/(2ρ)).
"""
function polytropic_pressure(γ, μ, ρ, E)
    return (γ - 1) * (E - μ^2 / (2 * ρ))
end

"""
    polytropic_flux(γ, μ, ρ, E, Σ)

Compute the 1D Euler flux with entropic pressure correction Σ.
Returns a tuple (fμ, fρ, fE).

State variables: μ = momentum, ρ = density, E = total energy.
Total pressure: P̄ = p + Σ where p is the polytropic pressure.
"""
function polytropic_flux(γ, μ, ρ, E, Σ)
    p = polytropic_pressure(γ, μ, ρ, E)
    p_bar = p + Σ

    fμ = μ^2 / ρ + p_bar
    fρ = μ
    fE = (E + p_bar) * μ / ρ

    return (fμ, fρ, fE)
end

"""
    sound_speed(γ, μ, ρ, E)

Compute the sound speed c = √(γ p / ρ).
"""
function sound_speed(γ, μ, ρ, E)
    p = polytropic_pressure(γ, μ, ρ, E)
    return sqrt(γ * p / ρ)
end

"""
    max_wavespeed(γ, μ, ρ, E)

Maximum wavespeed |u| + c for CFL computation.
"""
function max_wavespeed(γ, μ, ρ, E)
    u = μ / ρ
    c = sound_speed(γ, μ, ρ, E)
    return abs(u) + c
end

"""
    llf_flux(γ, μL, ρL, EL, ΣL, μR, ρR, ER, ΣR)

Local Lax-Friedrichs (Rusanov) numerical flux at an interface.
Returns a tuple (f★μ, f★ρ, f★E).
"""
function llf_flux(γ, μL, ρL, EL, ΣL, μR, ρR, ER, ΣR, β=1.0)
    # Wave speed estimate (β ≥ 1 increases interface dissipation)
    λ = β * max(max_wavespeed(γ, μL, ρL, EL), max_wavespeed(γ, μR, ρR, ER))

    # Left and right fluxes
    fμL, fρL, fEL = polytropic_flux(γ, μL, ρL, EL, ΣL)
    fμR, fρR, fER = polytropic_flux(γ, μR, ρR, ER, ΣR)

    # LLF: f* = (f_L + f_R)/2 + λ/2 (q_L - q_R)
    f_μ = (fμL + fμR) / 2 + λ * (μL - μR) / 2
    f_ρ = (fρL + fρR) / 2 + λ * (ρL - ρR) / 2
    f_E = (fEL + fER) / 2 + λ * (EL - ER) / 2

    return (f_μ, f_ρ, f_E)
end
