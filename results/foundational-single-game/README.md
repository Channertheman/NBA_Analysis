# Foundational single-game figures

These plainly named figures are exported from the saved outputs in
[`NBA_Analysis.ipynb`](../../NBA_Analysis.ipynb) and
[`QUT_Boostrap.ipynb`](../../QUT_Boostrap.ipynb). They cover Pacers at
Raptors on October 28, 2015 (game `0021500009`).

## Exploratory movement and scoring maps

| Figure | Description |
|---|---|
| [Non-scoring movement by quarter](./game-0021500009__movement-heatmaps__non-scoring-possessions__quarters-1-to-4.png) | Left-to-right-oriented ball density for non-scoring possessions |
| [Scoring movement by quarter](./game-0021500009__movement-heatmaps__scoring-possessions__quarters-1-to-4.png) | Left-to-right-oriented ball density for scoring possessions |
| [Conditional scoring probability by quarter](./game-0021500009__conditional-scoring-probability__minimum-sector-visits-3__quarters-1-to-4.png) | Descriptive scoring rate in sectors visited by at least three possessions |

## Initial QUT outputs

| Figure | Sampling configuration |
|---|---|
| [Full-game lambda distributions](./game-0021500009__qut-lambda-distributions__full-game__bootstraps-500.png) | 500 bootstraps |
| [Full-game pointwise coefficient intervals](./game-0021500009__qut-pointwise-coefficient-intervals__full-game__bootstraps-500.png) | 500 bootstraps |
| [Quarter 1 pointwise coefficient intervals](./game-0021500009__qut-pointwise-coefficient-intervals__quarter-1__bootstraps-250.png) | 250 bootstraps |
| [Quarter 2 pointwise coefficient intervals](./game-0021500009__qut-pointwise-coefficient-intervals__quarter-2__bootstraps-250.png) | 250 bootstraps |
| [Quarter 3 pointwise coefficient intervals](./game-0021500009__qut-pointwise-coefficient-intervals__quarter-3__bootstraps-250.png) | 250 bootstraps |
| [Quarter 4 pointwise coefficient intervals](./game-0021500009__qut-pointwise-coefficient-intervals__quarter-4__bootstraps-250.png) | 250 bootstraps |

Every QUT bootstrap samples 1,000 possessions and 100 tracking positions per
possession with replacement. These are historical single-game results and use
the legacy half-court-filtered input.

Run `python scripts/export_foundational_figures.py` from the repository root
to re-extract the nine PNGs without rerunning either notebook.
