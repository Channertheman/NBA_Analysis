# QUT profile recovery: 8 simulated games

This directory presents the profile-recovery results in names intended for
people browsing the project. Machine-generated checkpoints and caches remain
under `profile_sim_results/`.

## Run summary

- Source games: 10
- Simulated games: 8
- Possessions per simulated game: 92
- Baseline scoring probability: 10%
- Court grid: 20 by 10
- Profiles: reference, high-y-side, low-y-side, and perimeter
- Evaluation scopes: full game and quarters 1 through 4

All six detailed configurations are included because this study is already a selected validation: five settings from the ten-game ranking plus the 500-bootstrap comparison. The overview plots and the complete ranking tables are included.
Gray `NA` cells in overview plots denote configurations that were not run.

## Published detailed configurations

| Overall rank | Bootstraps | Possessions per bootstrap | Samples per possession | Selection reason |
|---:|---:|---:|---:|---|
| 1 | 50 | 1,000 | 250 | ten-game top-five validation |
| 2 | 100 | 1,000 | 250 | ten-game top-five validation |
| 3 | 100 | 1,000 | 200 | ten-game top-five validation |
| 4 | 500 | 1,000 | 100 | 500-bootstrap comparison |
| 5 | 50 | 2,500 | 250 | ten-game top-five validation |
| 6 | 100 | 2,500 | 250 | ten-game top-five validation |

## Directory guide

- `heatmaps/`: plots grouped first by spatial profile and then bootstrap count.
- `heatmap-index.csv`: searchable index of every published PNG.
- `rankings/overall.csv`: all completed configurations ranked across profiles.
- `rankings/by-profile.csv`: all completed configurations ranked within profile.

A detailed filename such as
`possessions-per-bootstrap-01000__samples-per-possession-0250.png` describes
the remaining two simulation counts. Its parent directory supplies the
bootstrap count, and this directory supplies the simulated-game count and
possessions per game.
