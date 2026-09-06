# QUT method-comparison figures

These figures are exported from the saved outputs in
[`projections.ipynb`](../../projections.ipynb). Each uses one shared color
scale to compare:

1. full-grid bounded slack;
2. full-grid Moore-Penrose projection;
3. reduced-grid bounded slack; and
4. reduced-grid Moore-Penrose projection.

The comparison uses 500 bootstraps, 1,000 possessions per bootstrap, 100
samples per possession, and a 20-by-10 court grid. An `X` marks a sector that
was removed from the reduced-grid fit.

| Game | Coefficient comparison |
|---|---|
| 0021500009 | [View PNG](./game-0021500009__full-and-reduced-grid__bounded-slack-vs-moore-penrose-projection.png) |
| 0021500390 | [View PNG](./game-0021500390__full-and-reduced-grid__bounded-slack-vs-moore-penrose-projection.png) |

Run `python scripts/export_projection_figures.py` from the repository root to
re-extract both PNGs from the notebook.
