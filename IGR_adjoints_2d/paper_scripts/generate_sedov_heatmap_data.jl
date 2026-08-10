#!/usr/bin/env julia
# ============================================================
# Generate heatmap images for the three-Sedov-blast figure.
#
# High-resolution 2D fields render poorly as native pgfplots
# surfaces, so each field is rasterized to a BARE PNG with
# CairoMakie (no axes, ticks, labels, colorbar, or margins) and
# then framed by a pgfplots axis in figures/tikz/sedov_heatmaps.tex
# (which supplies the frame, ticks, axis labels, and colorbar).
#
# Five panels, in the layout used by the figure:
#   primal  (forward at T):     u_x, p, Σ
#   adjoint (PDE adjoint at 0): a†_ux, a†_p
#
# Outputs (in figures/data/):
#   sedov_heatmap_<panel>.png   — bare rasterized field, one per panel
#   sedov_heatmap_defs.tex      — generated pgfplots colormap definitions
#                                 (sampled from Makie so the colorbar matches
#                                 the image exactly) and per-panel
#                                 \point-meta option macros
# ============================================================

using Printf, LinearAlgebra
using CairoMakie

include(joinpath(@__DIR__, "..", "IGRAdjoints2D.jl"))
using .IGRAdjoints2D

# ============================================================
# Output directory (paper repo)
# ============================================================
# Figure data is written here, inside this repository. Copy the contents of
# paper_data/ into the paper's figures/data/ directory to rebuild the figures.
const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper_data")
mkpath(DATA_DIR)

# ============================================================
# Problem setup — the same problem as the refinement study in
# generate_otd_refinement_sedov_data.jl.
# ============================================================
γ   = 1.4
L   = 1.0
p   = 4
N_e = 160
α   = 3.0 * (L / 20)^2   # FIXED physical IGR regularization — same problem as the OtD study
T   = 0.10
CFL = 0.4
n_iter = 60     # PCG iterations — converged elliptic solves (publication quality)
solver = :pcg

# Three blasts: centers and (slightly different) over-pressure amplitudes
blast_xc = [0.30, 0.70, 0.52]
blast_yc = [0.34, 0.40, 0.74]
blast_A  = [6.0,  5.0,  5.5]
σ_blast  = 0.09
p_bg     = 1.0
n_blast  = length(blast_A)

objective = :kinetic_energy

basis = DGBasis(p)
mesh  = CartesianMesh2D(N_e, N_e, L, L, basis)
n_p = p + 1

@printf("Sedov heatmap data: objective=%s, T=%.3f\n", objective, T)
@printf("  p=%d, N_e=%d×%d, α=%.4e, σ=%.2f, n_iter=%d, solver=%s\n",
        p, N_e, N_e, α, σ_blast, n_iter, solver)

# ============================================================
# Three-blast initial condition:  ρ=1, u=0, p = p_bg + Σ_k A_k exp(-r_k²/σ²)
# ============================================================
μx0 = zeros(size(mesh.x)); μy0 = zeros(size(mesh.x))
ρ0  = similar(mesh.x);     E0  = similar(mesh.x)
for idx in eachindex(mesh.x)
    x = mesh.x[idx]; y = mesh.y[idx]
    p_val = p_bg
    for k in 1:n_blast
        r2 = (x - blast_xc[k])^2 + (y - blast_yc[k])^2
        p_val += blast_A[k] * exp(-r2 / σ_blast^2)
    end
    ρ0[idx] = 1.0
    E0[idx] = p_val / (γ - 1)
end

# CFL-based time step
max_ws = 0.0
for idx in eachindex(μx0)
    global max_ws
    ws = max(max_wavespeed_x(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]),
             max_wavespeed_y(γ, μx0[idx], μy0[idx], ρ0[idx], E0[idx]))
    max_ws = max(max_ws, ws)
end
Δt = CFL * min(mesh.Δx, mesh.Δy) / ((2*p + 1) * max_ws)
n_steps = ceil(Int, T / Δt)
Δt = T / n_steps
@printf("  n_steps=%d, Δt=%.4e\n", n_steps, Δt)

# ============================================================
# Forward solve (store) + backward PDE adjoint
# ============================================================
println("\nForward solve (storing snapshots)...")
snapshots, n_steps_fwd = run_forward_store(μx0, μy0, ρ0, E0, Δt, T,
    basis, mesh, γ, α; n_iter=n_iter, solver=solver)

println("PDE adjoint (backward integration)...")
aμx_pde, aμy_pde, aρ_pde, aE_pde = run_adjoint_conservative(snapshots, n_steps_fwd,
    Δt, T, basis, mesh, γ, α; n_iter=n_iter, objective=objective, solver=solver)

# Conservative → primitive L² adjoints (ux, uy, ρ, p)
ux0 = μx0 ./ ρ0
uy0 = μy0 ./ ρ0
pde_ux = ρ0 .* aμx_pde .+ ρ0 .* ux0 .* aE_pde
pde_uy = ρ0 .* aμy_pde .+ ρ0 .* uy0 .* aE_pde
pde_ρ  = ux0 .* aμx_pde .+ uy0 .* aμy_pde .+ aρ_pde .+ (ux0.^2 .+ uy0.^2)./2 .* aE_pde
pde_p  = aE_pde ./ (γ - 1)

# Forward solution at T (primitive variables)
μxT, μyT, ρT, ET, ΣT = snapshots[end]
uxT = μxT ./ ρT
uyT = μyT ./ ρT
pT  = (γ - 1) .* (ET .- (μxT.^2 .+ μyT.^2) ./ (2 .* ρT))   # mechanical pressure

# ============================================================
# Flatten tensor-product DG DOFs to a logical global grid
# ============================================================
function flatten2d(field)
    Q = zeros(N_e * n_p, N_e * n_p)
    for ey in 1:N_e, ex in 1:N_e, j in 1:n_p, i in 1:n_p
        Q[(ex-1)*n_p + i, (ey-1)*n_p + j] = field[i,j,ex,ey]
    end
    return Q
end
xg = [mesh.x[i,1,ex,1] for ex in 1:N_e for i in 1:n_p]
yg = [mesh.y[1,j,1,ey] for ey in 1:N_e for j in 1:n_p]
xg .+= range(0, 1e-9, length=length(xg))
yg .+= range(0, 1e-9, length=length(yg))

# ============================================================
# Bare-image renderer: a single heatmap filling the whole canvas,
# no decorations / spines / margins, so a pgfplots axis can frame it.
# ============================================================
const NPX = 1200   # pixels per side
function save_bare_heatmap(path, Q, cmap, crange)
    fig = Figure(size=(NPX, NPX), figure_padding=0)
    ax  = Axis(fig[1, 1])
    heatmap!(ax, xg, yg, Q; colormap=cmap, colorrange=crange)
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, 0, L); ylims!(ax, 0, L)
    save(path, fig)
end

# Color range helper: symmetric about 0 for diverging fields, [min,max] otherwise.
crange(Q, diverging) = diverging ? (m = maximum(abs, Q); (-m, m)) : (minimum(Q), maximum(Q))

# Custom colormaps matching the paper palette (figure_style.tex):
#   sign-indefinite (diverging): steelblue → white → orange  (white = 0)
#   positive (sequential):       white → orange
const STEELBLUE = RGBf(0xA1/255, 0xBD/255, 0xC7/255)
const ORANGE    = RGBf(0xD9/255, 0x8C/255, 0x21/255)
const WHITE     = RGBf(1, 1, 1)
const CMAP_DIV  = cgrad([STEELBLUE, WHITE, ORANGE])   # diverging, centered on white
const CMAP_POS  = cgrad([WHITE, ORANGE])              # sequential, positive
const PGF_DIV   = "divSteelOrange"                     # matching pgfplots colormap names
const PGF_POS   = "posWhiteOrange"

# Σ is sign-indefinite in principle but predominantly positive at the
# fronts; pick the colormap from the actual data.
Σ_flat = flatten2d(ΣT)
Σ_div  = minimum(Σ_flat) < -0.02 * maximum(Σ_flat)

# Panel table: (key, field, makie colormap, pgfplots colormap name, diverging?)
panels = [
    ("fwd_ux",    flatten2d(uxT),    CMAP_DIV, PGF_DIV, true),
    ("fwd_p",     flatten2d(pT),     CMAP_POS, PGF_POS, false),
    ("fwd_sigma", Σ_flat,            Σ_div ? CMAP_DIV : CMAP_POS,
                                     Σ_div ? PGF_DIV  : PGF_POS,  Σ_div),
    ("adj_ux",    flatten2d(pde_ux), CMAP_DIV, PGF_DIV, true),
    ("adj_p",     flatten2d(pde_p),  CMAP_DIV, PGF_DIV, true),
]

# macro-name suffix per panel key (LaTeX macros: letters only)
const MACRO = Dict(
    "fwd_ux"=>"FwdUx", "fwd_uy"=>"FwdUy", "fwd_p"=>"FwdP", "fwd_sigma"=>"FwdSigma",
    "adj_ux"=>"AdjUx", "adj_uy"=>"AdjUy", "adj_p"=>"AdjP")


# ============================================================
# Render the bare PNGs and record per-panel ranges
# ============================================================
println("\nRendering bare heatmap PNGs...")
ranges = Dict{String,Tuple{Float64,Float64}}()
mapname = Dict{String,String}()
for (key, Q, cmap, pgfname, diverging) in panels
    cr = crange(Q, diverging)
    ranges[key]  = cr
    mapname[key] = pgfname
    png = joinpath(DATA_DIR, "sedov_heatmap_$(key).png")
    save_bare_heatmap(png, Q, cmap, cr)
    @printf("  %-10s  range=[% .4e, % .4e]  → %s\n", key, cr[1], cr[2], basename(png))
end

# ============================================================
# Emit pgfplots colormap definitions (sampled from Makie) + per-panel
# option macros, so the colorbar matches each image exactly.
# ============================================================
function write_colormap(io, name, cmap; n=64)
    cs = to_colormap(cmap)
    m  = length(cs)
    println(io, "\\pgfplotsset{colormap={$name}{")
    for k in 0:n-1
        t = k / (n - 1)
        idx = clamp(round(Int, 1 + t * (m - 1)), 1, m)
        c = cs[idx]
        @printf(io, "  rgb=(%.5f,%.5f,%.5f)\n", c.r, c.g, c.b)
    end
    println(io, "}}")
end

defs = joinpath(DATA_DIR, "sedov_heatmap_defs.tex")
open(defs, "w") do io
    println(io, "% GENERATED by paper_scripts/generate_sedov_heatmap_data.jl — do not edit by hand.")
    println(io, "% pgfplots colormaps sampled from the CairoMakie colormaps (paper palette:")
    println(io, "% steelblue-white-orange diverging, white-orange sequential) so the colorbar")
    println(io, "% matches the PNGs exactly,")
    println(io, "% plus one option macro per panel carrying its colormap name and color limits.")
    println(io)
    write_colormap(io, PGF_DIV, CMAP_DIV)
    write_colormap(io, PGF_POS, CMAP_POS)
    println(io)
    for (key, _, _, _, _) in panels
        cmin, cmax = ranges[key]
        @printf(io, "\\def\\sedovOpts%s{colormap name=%s, point meta min=%.6e, point meta max=%.6e}\n",
                MACRO[key], mapname[key], cmin, cmax)
    end
end

println("\nWrote:")
println("  8 PNGs           → $DATA_DIR/sedov_heatmap_*.png")
println("  colormap+limits  → $defs")
println("Figure layout: figures/tikz/sedov_heatmaps.tex (\\input's the defs).")
