# ============================================================
# 2D Euler equations: EOS, directional fluxes, sound speed,
# numerical fluxes. All functions are type-generic (no ::Float64).
#
# Conserved state ordering throughout the package: (μx, μy, ρ, E)
#   μx, μy = momentum components
#   ρ      = density
#   E      = total energy
# ============================================================

export polytropic_pressure, polytropic_flux_x, polytropic_flux_y,
       sound_speed, max_wavespeed_x, max_wavespeed_y,
       llf_flux_x, llf_flux_y

"""
    polytropic_pressure(γ, μx, μy, ρ, E)

Polytropic (ideal gas) pressure: P = (γ-1)(E - |μ|²/(2ρ)).
"""
function polytropic_pressure(γ, μx, μy, ρ, E)
    return (γ - 1) * (E - (μx^2 + μy^2) / (2 * ρ))
end

"""
    polytropic_flux_x(γ, μx, μy, ρ, E, Σ)

x-component of the 2D Euler flux with entropic pressure correction Σ.
Total pressure P̄ = P + Σ enters the momentum and energy fluxes.
Returns (fμx, fμy, fρ, fE).
"""
function polytropic_flux_x(γ, μx, μy, ρ, E, Σ)
    P = polytropic_pressure(γ, μx, μy, ρ, E)
    p_bar = P + Σ
    ux = μx / ρ

    fμx = μx * ux + p_bar
    fμy = μy * ux
    fρ  = μx
    fE  = (E + p_bar) * ux

    return (fμx, fμy, fρ, fE)
end

"""
    polytropic_flux_y(γ, μx, μy, ρ, E, Σ)

y-component of the 2D Euler flux with entropic pressure correction Σ.
Returns (fμx, fμy, fρ, fE).
"""
function polytropic_flux_y(γ, μx, μy, ρ, E, Σ)
    P = polytropic_pressure(γ, μx, μy, ρ, E)
    p_bar = P + Σ
    uy = μy / ρ

    fμx = μx * uy
    fμy = μy * uy + p_bar
    fρ  = μy
    fE  = (E + p_bar) * uy

    return (fμx, fμy, fρ, fE)
end

"""
    sound_speed(γ, μx, μy, ρ, E)

Sound speed c = √(γ P / ρ).
"""
function sound_speed(γ, μx, μy, ρ, E)
    P = polytropic_pressure(γ, μx, μy, ρ, E)
    return sqrt(γ * P / ρ)
end

"""
    max_wavespeed_x(γ, μx, μy, ρ, E)

Maximum x-direction wavespeed |ux| + c for CFL / LLF dissipation.
"""
function max_wavespeed_x(γ, μx, μy, ρ, E)
    return abs(μx / ρ) + sound_speed(γ, μx, μy, ρ, E)
end

"""
    max_wavespeed_y(γ, μx, μy, ρ, E)

Maximum y-direction wavespeed |uy| + c for CFL / LLF dissipation.
"""
function max_wavespeed_y(γ, μx, μy, ρ, E)
    return abs(μy / ρ) + sound_speed(γ, μx, μy, ρ, E)
end

"""
    llf_flux_x(γ, μxL,μyL,ρL,EL,ΣL, μxR,μyR,ρR,ER,ΣR, β=1.0)

Local Lax-Friedrichs (Rusanov) numerical flux across an x-normal face.
Returns (f★μx, f★μy, f★ρ, f★E).
"""
function llf_flux_x(γ, μxL, μyL, ρL, EL, ΣL,
                       μxR, μyR, ρR, ER, ΣR, β=1.0)
    λ = β * max(max_wavespeed_x(γ, μxL, μyL, ρL, EL),
                max_wavespeed_x(γ, μxR, μyR, ρR, ER))

    fμxL, fμyL, fρL, fEL = polytropic_flux_x(γ, μxL, μyL, ρL, EL, ΣL)
    fμxR, fμyR, fρR, fER = polytropic_flux_x(γ, μxR, μyR, ρR, ER, ΣR)

    f_μx = (fμxL + fμxR) / 2 + λ * (μxL - μxR) / 2
    f_μy = (fμyL + fμyR) / 2 + λ * (μyL - μyR) / 2
    f_ρ  = (fρL  + fρR)  / 2 + λ * (ρL  - ρR)  / 2
    f_E  = (fEL  + fER)  / 2 + λ * (EL  - ER)  / 2

    return (f_μx, f_μy, f_ρ, f_E)
end

"""
    llf_flux_y(γ, μxL,μyL,ρL,EL,ΣL, μxR,μyR,ρR,ER,ΣR, β=1.0)

Local Lax-Friedrichs (Rusanov) numerical flux across a y-normal face.
Returns (f★μx, f★μy, f★ρ, f★E).
"""
function llf_flux_y(γ, μxL, μyL, ρL, EL, ΣL,
                       μxR, μyR, ρR, ER, ΣR, β=1.0)
    λ = β * max(max_wavespeed_y(γ, μxL, μyL, ρL, EL),
                max_wavespeed_y(γ, μxR, μyR, ρR, ER))

    fμxL, fμyL, fρL, fEL = polytropic_flux_y(γ, μxL, μyL, ρL, EL, ΣL)
    fμxR, fμyR, fρR, fER = polytropic_flux_y(γ, μxR, μyR, ρR, ER, ΣR)

    f_μx = (fμxL + fμxR) / 2 + λ * (μxL - μxR) / 2
    f_μy = (fμyL + fμyR) / 2 + λ * (μyL - μyR) / 2
    f_ρ  = (fρL  + fρR)  / 2 + λ * (ρL  - ρR)  / 2
    f_E  = (fEL  + fER)  / 2 + λ * (EL  - ER)  / 2

    return (f_μx, f_μy, f_ρ, f_E)
end
