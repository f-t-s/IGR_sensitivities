# paper_data

Output directory for the reproduction scripts. Every script in
`IGR_adjoints_1d/paper_scripts/` and `IGR_adjoints_2d/paper_scripts/` writes its
CSVs, rasterized heatmap PNGs, and generated TikZ colormap definitions here.

This directory ships empty: the figure data is regenerated rather than
distributed, and its contents are gitignored apart from this file.

The layout is flat and matches the paper's `figures/data/` directory one-to-one, so
rebuilding the figures is a straight copy:

```
cp paper_data/* /path/to/paper/figures/data/
```

The 1D convergence ablation also *reads* from this directory: it post-processes the
`pde_mesh_sweep_*_primal.csv` and `pde_mesh_sweep_*_adj_pde.csv` files written by the
mesh sweep, so the sweep has to run first (the runner
`IGR_adjoints_1d/run_all_experiments.jl` already orders them correctly).
