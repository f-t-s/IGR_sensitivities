#!/usr/bin/env julia
# ============================================================
# Reproduce every 1D figure dataset used in the paper.
#
#   julia --project=. run_all_experiments.jl            # everything
#   julia --project=. run_all_experiments.jl shift      # only matching scripts
#   julia --project=. run_all_experiments.jl --list     # show plan and exit
#
# Each generator runs in its own Julia process, so memory is released
# between experiments and one failure cannot corrupt the next run.
# Order matters: the convergence ablation post-processes the CSVs
# written by the mesh sweep, so it must run last.
# ============================================================

using Printf

const ROOT = @__DIR__

# (script, what it produces, rough serial runtime on an M-series laptop)
const PLAN = [
    ("generate_shift_sensitivity_data.jl",
     "forward sensitivity w.r.t. cyclic shift  -> fig:shift_postshock",
     "~3 min"),
    ("generate_gamma_sensitivity_data.jl",
     "forward sensitivity w.r.t. gamma         -> fig:blast_gamma",
     "~3 min"),
    ("generate_pde_vs_enzyme_alpha_data.jl",
     "PDE adjoint vs Enzyme, 3 alpha ratios    -> fig:pde_vs_enzyme_adjoint",
     "~30 min"),
    ("generate_pde_adjoint_mesh_sweep_data.jl",
     "anchored-pair mesh sweep, N_e 128..4096  -> fig:pde_mesh_sweep",
     "~7.5 h"),
    ("generate_adjoint_convergence_ablation.jl",
     "L1 ablation (reads the sweep CSVs above) -> fig:adjoint_convergence_ablation",
     "~1 min"),
]

function main()
    args   = filter(a -> a != "--list", ARGS)
    listing = "--list" in ARGS
    plan   = isempty(args) ? PLAN :
             filter(e -> any(occursin(a, e[1]) for a in args), PLAN)

    if isempty(plan)
        println("No scripts match $(args). Available:")
        for (s, _, _) in PLAN; println("  ", s); end
        return 1
    end

    println("="^72)
    println("1D reproduction plan ($(length(plan)) script(s))")
    println("="^72)
    for (s, what, cost) in plan
        @printf("  %-42s %-8s %s\n", s, cost, what)
    end
    println()
    listing && return 0

    results = Tuple{String,Bool,Float64}[]
    for (i, (s, _, cost)) in enumerate(plan)
        println("="^72)
        @printf("[%d/%d] %s   (expected %s)\n", i, length(plan), s, cost)
        println("="^72)
        flush(stdout)
        path = joinpath(ROOT, "paper_scripts", s)
        t0 = time()
        ok = try
            run(`$(Base.julia_cmd()) --project=$ROOT $path`)
            true
        catch
            false
        end
        push!(results, (s, ok, time() - t0))
        ok || @warn "FAILED: $s (continuing with the remaining scripts)"
    end

    println("\n", "="^72)
    println("Summary")
    println("="^72)
    for (s, ok, dt) in results
        @printf("  %-8s %-42s %7.1f min\n", ok ? "ok" : "FAILED", s, dt / 60)
    end
    total = sum(r -> r[3], results) / 3600
    @printf("  total: %.2f h\n", total)
    nfail = count(r -> !r[2], results)
    if nfail > 0
        println("\n$nfail script(s) failed.")
        return 1
    end
    println("\nAll 1D figure data regenerated.")
    return 0
end

exit(main())
