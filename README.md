# IGR adjoints — code accompanying the paper

Reference implementation and reproduction scripts for *Information geometric
regularization for sensitivities of flows with shocks*.

Two self-contained Julia packages, one per spatial dimension:

```
IGR_adjoints_1d/         IGR_adjoints_2d/
  IGRAdjoints1D.jl         IGRAdjoints2D.jl     module entry point (includes the files below)
  basis.jl                 basis.jl             nodal Gauss–Lobatto DG basis
  mesh.jl                  mesh.jl              periodic mesh
  euler.jl                 euler.jl             equation of state, wave speeds, LLF flux
  hyperbolic.jl            hyperbolic.jl        DG semi-discretization of the hyperbolic part
  elliptic.jl              elliptic.jl          matrix-free SIP solver for the entropic pressure Σ
  forward.jl               forward.jl           IGR forward solve (SSP-RK3, elliptic solve per stage)
  sensitivity.jl           sensitivity.jl       forward (tangent) sensitivity equations
  adjoint_pde.jl           adjoint_pde.jl       continuous adjoint PDE, conservative form
  enzyme_adjoint.jl        enzyme_adjoint.jl    discrete adjoint via Enzyme reverse mode
  initial_conditions.jl    initial_conditions.jl
  paper_scripts/           paper_scripts/       one script per figure (see below)
  tests/                   tests/               verification tests

paper_data/                                     output directory for all figure data
```

## Requirements

Julia 1.12.1 (the version recorded in both `Manifest.toml` files). From either
package directory:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

No GPU, threading, or display is required. The 1D package depends on Enzyme,
ForwardDiff and FastGaussQuadrature, the 2D package on Enzyme,
FastGaussQuadrature and CairoMakie (used only to rasterize the heatmap panels).

## Reproducing the figures

Each script writes its output into `paper_data/` at the root of this repository.
That directory mirrors the paper's `figures/data/` one-to-one, so rebuilding the
figures from a fresh run is a straight copy:

```
cp paper_data/* /path/to/paper/figures/data/
```

To regenerate everything for one dimension, use the runners, which launch each
generator in its own Julia process (so memory is released between experiments),
order them correctly, and print a timing summary:

```
cd IGR_adjoints_1d && julia --project=. run_all_experiments.jl
cd IGR_adjoints_2d && julia --project=. run_all_experiments.jl
```

They accept a substring to run a subset, and `--list` for a dry run:

```
julia --project=. run_all_experiments.jl shift      # just that experiment
julia --project=. run_all_experiments.jl --list     # show the plan, run nothing
```

Individual scripts can also be run directly from the package directory, e.g.

```
cd IGR_adjoints_1d && julia --project=. paper_scripts/generate_shift_sensitivity_data.jl
```

| Paper figure | Script | Approx. runtime |
|---|---|---|
| Cyclic-shift forward sensitivity | `1d/generate_shift_sensitivity_data.jl` | ~1 min |
| γ-sensitivity of interacting blasts | `1d/generate_gamma_sensitivity_data.jl` | ~2 min |
| Continuous vs. discrete (Enzyme) adjoint | `1d/generate_pde_vs_enzyme_alpha_data.jl` | ~20 min |
| Adjoints across mesh resolutions | `1d/generate_pde_adjoint_mesh_sweep_data.jl` | ~8 h (dominated by `N_e = 4096`) |
| Linear vs. sublinear α-scaling | `1d/generate_adjoint_convergence_ablation.jl` | seconds (post-processes the sweep CSVs) |
| 2D triple Sedov blast | `2d/generate_sedov_heatmap_data.jl` + `2d/generate_otd_refinement_sedov_data.jl` | ~1.5 h + ~30 min |
| 2D blast–vortex interaction | `2d/generate_oseen_heatmap_data.jl` + `2d/generate_otd_refinement_oseen_data.jl` | ~2.5 h + ~2 h |

The ablation script consumes the `pde_mesh_sweep_*_primal.csv` and `*_adj_pde.csv`
files produced by the mesh sweep, so run the sweep first.

## Numerical settings

All experiments use a fixed number of elliptic iterations rather than a
convergence tolerance: this keeps the discrete forward map an explicit
differentiable composition, so that automatic differentiation through it is well
defined. The solver and iteration count differ per experiment (damped Jacobi with
50 iterations for the forward-sensitivity figures, Chebyshev-accelerated Jacobi
with 5 for the Enzyme comparison, diagonally preconditioned CG with 102 in 1D and
60 in 2D for the adjoint studies); each script states its own settings at the top.

## Tests

```
julia --project=. tests/run_euler_test.jl              # forward solver
julia --project=. tests/run_sensitivity_test.jl        # forward sensitivities vs. AD/FD
julia --project=. tests/run_adjoint_conservative_test.jl   # 1D adjoint vs. Enzyme  (2d: run_adjoint_test.jl)
julia --project=. tests/run_enzyme_test.jl             # Enzyme gradients vs. finite differences
julia --project=. tests/run_quasi1d_test.jl            # 2d only: quasi-1D run reproduces the 1D package
```

`run_adjoint_test.jl` (2D) is the sharpest check on the derivation: it compares
the continuous adjoint against Enzyme reverse mode on a Taylor–Green case and
asserts that the disagreement stays within the discretize-then-differentiate gap
and does not grow under refinement. A sign or derivation error in
`adjoint_pde.jl` produces an O(1) failure instead. `run_quasi1d_test.jl`
complements it by cross-validating the two independent solvers against each
other: a 2D run on y-uniform data with zero transverse momentum must reproduce
the 1D package to roundoff.
