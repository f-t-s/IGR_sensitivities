#!/usr/bin/env julia
# ============================================================
# Layer-resolution ablation for the PDE adjoint mesh sweep.
#
# Reads the mesh-sweep CSVs for the two anchored pairs of scaling designs
#   lin2: √α = 2Δx               gu2p: √α = 2·Δx₀^{1/3}Δx^{2/3}
#   lin4: √α = 4Δx               gu4p: √α = 4·Δx₀^{1/3}Δx^{2/3}
# (each pair shares its α at N_e = 128, Δx₀ = L/128)
# and computes, for the primal velocity u(·,T) and the adjoint
# velocity a†_u(·,0), the L¹ norm of the difference between
# solutions at consecutive resolutions (sampled on a uniform
# 20 000-point grid).
#
# Output (in figures/data/):
#   adjoint_convergence_ablation.csv
#     — Ne, prim_lin2, prim_gu2p, prim_lin4, prim_gu4p, adj_lin2, adj_gu2p, adj_lin4, adj_gu4p
#   (Ne is the finer resolution of each pair; missing pairs are
#    written as nan and skipped by pgfplots.)
# ============================================================

using DelimitedFiles, Printf

# Figure data is written here, inside this repository. Copy the contents of
# paper_data/ into the paper's figures/data/ directory to rebuild the figures.
const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper_data")

function load_curve(file)
    d = readdlm(file, ',', skipstart=1)
    x = Float64[]; u = Float64[]
    for i in axes(d, 1)
        xi, ui = d[i, 1], d[i, 2]
        (xi isa Number && ui isa Number && isfinite(xi) && isfinite(ui)) || continue
        push!(x, xi); push!(u, ui)
    end
    p = sortperm(x)
    x[p], u[p]
end

function interp(x, u, xq)
    j = clamp(searchsortedlast(x, xq), 1, length(x) - 1)
    t = (xq - x[j]) / (x[j+1] - x[j])
    u[j] * (1 - t) + u[j+1] * t
end

const XG = collect(range(0.001, 0.999, length=20000))
on_grid(file) = let (x, u) = load_curve(file)
    [interp(x, u, q) for q in XG]
end
l1(a, b) = sum(abs, a .- b) / length(a)

# (label, file prefix, resolutions available)
const DESIGNS = [
    ("lin2", "pde_mesh_sweep_lin2", [128, 256, 512, 1024, 2048, 4096]),
    ("gu2p", "pde_mesh_sweep_gu2p", [128, 256, 512, 1024, 2048, 4096]),
    ("lin4", "pde_mesh_sweep_lin4", [128, 256, 512, 1024, 2048, 4096]),
    ("gu4p", "pde_mesh_sweep_gu4p", [128, 256, 512, 1024, 2048, 4096]),
]
const NE_PAIRS = [256, 512, 1024, 2048, 4096]   # finer level of each pair

diffs = Dict{Tuple{String,String,Int},Float64}()
for (label, prefix, Ns) in DESIGNS
    for k in 1:length(Ns)-1
        Nc, Nf = Ns[k], Ns[k+1]
        for (kind, suffix) in (("prim", "primal"), ("adj", "adj_pde"))
            a = on_grid(joinpath(DATA_DIR, "$(prefix)_Ne$(Nc)_$(suffix).csv"))
            b = on_grid(joinpath(DATA_DIR, "$(prefix)_Ne$(Nf)_$(suffix).csv"))
            diffs[(kind, label, Nf)] = l1(a, b)
        end
        @printf("%s %-4s  %4d vs %4d done\n", prefix, "", Nc, Nf)
    end
end

out = joinpath(DATA_DIR, "adjoint_convergence_ablation.csv")
open(out, "w") do io
    println(io, "Ne,prim_lin2,prim_gu2p,prim_lin4,prim_gu4p,adj_lin2,adj_gu2p,adj_lin4,adj_gu4p")
    for Nf in NE_PAIRS
        vals = [get(diffs, (kind, label, Nf), NaN)
                for kind in ("prim", "adj"), label in ("lin2", "gu2p", "lin4", "gu4p")]
        @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n", Nf,
                vals[1,1], vals[1,2], vals[1,3], vals[1,4], vals[2,1], vals[2,2], vals[2,3], vals[2,4])
    end
end
println("Wrote $out")
