#!/usr/bin/env julia
# ============================================================
# Reproduce every 2D figure dataset used in the paper.
#
#   julia --project=. run_all_experiments.jl            # everything
#   julia --project=. run_all_experiments.jl shift      # only matching scripts
#   julia --project=. run_all_experiments.jl --list     # show plan and exit
#
# Each generator runs in its own Julia process, so memory is released
# between experiments and one failure cannot corrupt the next run.
# The scripts are independent; the heatmap runs at N_e = 160 store the
# full forward trajectory and need on the order of 40 GB of memory
# (the OtD refinement runs are far smaller).
# ============================================================

using Printf

const ROOT = @__DIR__

# (script, what it produces, rough serial runtime on an M-series laptop)
const PLAN = [
    ("generate_sedov_heatmap_data.jl",
     "triple Sedov blast, primal + adjoint fields -> fig:sedov_heatmaps_2d",
     "~2.5 h"),
    ("generate_otd_refinement_sedov_data.jl",
     "adjoint vs FD under refinement (Sedov)      -> fig:sedov_heatmaps_2d",
     "~40 min"),
    ("generate_oseen_heatmap_data.jl",
     "blast-vortex interaction, primal + adjoint  -> fig:oseen_heatmaps_2d",
     "~3 h"),
    ("generate_otd_refinement_oseen_data.jl",
     "adjoint vs FD under refinement (Oseen)      -> fig:oseen_heatmaps_2d",
     "~2 h"),
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
    println("2D reproduction plan ($(length(plan)) script(s))")
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
    println("\nAll 2D figure data regenerated.")
    return 0
end

exit(main())
