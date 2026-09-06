# QUT profile-recovery results

These are the curated, audience-facing outputs from the profile-simulation
study. Results are separated by simulated-game count, and every numerical
component of a path is labeled with what it counts.

| Study | Possessions per game | Detailed heatmaps | Overview heatmaps |
|---|---:|---:|---:|
| [8 simulated games](./simulated-games-08__possessions-per-game-92/) | 92 | 24 | 12 |
| [10 simulated games](./simulated-games-10__possessions-per-game-92/) | 92 | 24 | 12 |

Each study directory contains complete CSV rankings, a searchable heatmap
index, run documentation, and selected PNGs. The full generated collections
and resumable simulation artifacts remain local under `profile_sim_results/`.

## Profile names

- `reference`: the original spatial scoring profile.
- `high-y-side`: the shifted profile on the higher-y side of the court.
- `low-y-side`: the shifted profile on the lower-y side of the court.
- `perimeter`: the perimeter-focused spatial profile.

## Overview metrics

- `support-region-similarity`: similarity within the intended profile region.
- `penalized-profile-similarity`: similarity penalized for recovered signal
  outside the intended profile.
- `off-profile-energy`: the recovered fraction outside the intended profile.
