"""Export curated, human-readable QUT profile-recovery results.

The simulation notebook keeps resumable and machine-oriented artifacts under
``profile_sim_results``. This script publishes a smaller, stable presentation
layer under ``results/qut-profile-recovery`` without changing those artifacts.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "qut-profile-recovery"
TOP_OVERALL_CONFIGURATIONS = 5


@dataclass(frozen=True)
class Collection:
    simulated_games: int
    possessions_per_game: int
    source_directory: Path
    include_all_configurations: bool = False

    @property
    def public_name(self) -> str:
        return (
            f"simulated-games-{self.simulated_games:02d}__"
            f"possessions-per-game-{self.possessions_per_game}"
        )


COLLECTIONS = (
    Collection(
        simulated_games=8,
        possessions_per_game=92,
        include_all_configurations=True,
        source_directory=PROJECT_ROOT
        / "profile_sim_results"
        / "heatmaps"
        / "qut_8games__completed__base10pct__B50-100-500",
    ),
    Collection(
        simulated_games=10,
        possessions_per_game=92,
        source_directory=PROJECT_ROOT
        / "profile_sim_results"
        / "heatmaps"
        / "qut_completed_runs__base10pct__B10-50-100-500",
    ),
)

PROFILE_NAMES = {
    "reference": "reference",
    "high_y_side": "high-y-side",
    "low_y_side": "low-y-side",
    "perimeter": "perimeter",
}

OVERVIEW_NAMES = {
    "support_cosine": "support-region-similarity.png",
    "penalized_profile_cosine": "penalized-profile-similarity.png",
    "outside_energy_fraction": "off-profile-energy.png",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def as_int(value: str) -> int:
    return int(float(value))


def configuration(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        as_int(row["n_bootstraps"]),
        as_int(row["possessions_per_bootstrap"]),
        as_int(row["samples_per_possession"]),
    )


def portable_source_path(collection: Collection, relative_path: str) -> Path:
    """Resolve manifest paths written on Windows on any operating system."""
    return collection.source_directory.joinpath(*PureWindowsPath(relative_path).parts)


def select_configurations(
    overall_ranking: list[dict[str, str]],
    include_all_configurations: bool,
) -> tuple[list[dict[str, str]], set[tuple[int, int, int]]]:
    selected_rows: list[dict[str, str]] = []
    selected_configurations: set[tuple[int, int, int]] = set()

    for row in overall_ranking:
        is_top_overall = as_int(row["performance_rank"]) <= TOP_OVERALL_CONFIGURATIONS
        is_bootstrap_comparison = as_int(row["n_bootstraps"]) == 500
        row_configuration = configuration(row)
        if (
            include_all_configurations
            or is_top_overall
            or is_bootstrap_comparison
        ) and (
            row_configuration not in selected_configurations
        ):
            selected_rows.append(row)
            selected_configurations.add(row_configuration)

    return selected_rows, selected_configurations


def detailed_destination(
    public_directory: Path,
    profile: str,
    n_bootstraps: int,
    possessions_per_bootstrap: int,
    samples_per_possession: int,
) -> Path:
    return (
        public_directory
        / "heatmaps"
        / PROFILE_NAMES[profile]
        / f"bootstrap-count-{n_bootstraps:04d}"
        / (
            f"possessions-per-bootstrap-{possessions_per_bootstrap:05d}__"
            f"samples-per-possession-{samples_per_possession:04d}.png"
        )
    )


def overview_destination(
    public_directory: Path, profile: str, metric: str
) -> Path:
    return (
        public_directory
        / "heatmaps"
        / PROFILE_NAMES[profile]
        / "overview"
        / OVERVIEW_NAMES[metric]
    )


def copy_image(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Expected heatmap does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_heatmap_index(
    destination: Path, rows: list[dict[str, str | int]]
) -> None:
    fieldnames = (
        "image_type",
        "profile",
        "metric",
        "bootstrap_count",
        "possessions_per_bootstrap",
        "samples_per_possession",
        "relative_path",
    )
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_collection_readme(
    collection: Collection,
    public_directory: Path,
    selected_rows: list[dict[str, str]],
) -> None:
    selection_table = []
    for row in selected_rows:
        rank = as_int(row["performance_rank"])
        if collection.include_all_configurations:
            reason = "ten-game top-five validation"
        else:
            reason = "top-five overall"
        if as_int(row["n_bootstraps"]) == 500:
            reason = "500-bootstrap comparison"
        selection_table.append(
            "| "
            + " | ".join(
                (
                    str(rank),
                    str(as_int(row["n_bootstraps"])),
                    f"{as_int(row['possessions_per_bootstrap']):,}",
                    f"{as_int(row['samples_per_possession']):,}",
                    reason,
                )
            )
            + " |"
        )

    if collection.include_all_configurations:
        curation_note = (
            "All six detailed configurations are included because this study is "
            "already a selected validation: five settings from the ten-game "
            "ranking plus the 500-bootstrap comparison."
        )
    else:
        curation_note = (
            "The detailed heatmaps are a curated export: the five highest-ranked "
            "overall configurations plus the 500-bootstrap comparison."
        )

    readme = f"""# QUT profile recovery: {collection.simulated_games} simulated games

This directory presents the profile-recovery results in names intended for
people browsing the project. Machine-generated checkpoints and caches remain
under `profile_sim_results/`.

## Run summary

- Source games: 10
- Simulated games: {collection.simulated_games}
- Possessions per simulated game: {collection.possessions_per_game}
- Baseline scoring probability: 10%
- Court grid: 20 by 10
- Profiles: reference, high-y-side, low-y-side, and perimeter
- Evaluation scopes: full game and quarters 1 through 4

{curation_note} The overview plots and the complete ranking tables are included.
Gray `NA` cells in overview plots denote configurations that were not run.

## Published detailed configurations

| Overall rank | Bootstraps | Possessions per bootstrap | Samples per possession | Selection reason |
|---:|---:|---:|---:|---|
{chr(10).join(selection_table)}

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
"""
    (public_directory / "README.md").write_text(readme, encoding="utf-8")


def export_collection(collection: Collection) -> tuple[int, int]:
    manifest_path = collection.source_directory / "plot_manifest.csv"
    ranking_directory = collection.source_directory / "summaries"
    overall_source = ranking_directory / "performance_ranking_overall.csv"
    by_profile_source = ranking_directory / "performance_ranking_by_profile.csv"
    required_files = (manifest_path, overall_source, by_profile_source)
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing required source files:\n{missing}")

    manifest = read_csv(manifest_path)
    overall_ranking = read_csv(overall_source)
    selected_rows, selected_configurations = select_configurations(
        overall_ranking, collection.include_all_configurations
    )

    public_directory = RESULTS_ROOT / collection.public_name
    if public_directory.exists():
        shutil.rmtree(public_directory)
    public_directory.mkdir(parents=True)

    index_rows: list[dict[str, str | int]] = []
    detailed_count = 0
    overview_count = 0

    for row in manifest:
        image_type = row["plot_type"]
        profile = row["profile"]
        if profile not in PROFILE_NAMES:
            raise ValueError(f"Unknown profile in heatmap manifest: {profile}")

        source = portable_source_path(collection, row["relative_path"])
        if image_type == "detailed_profile":
            row_configuration = configuration(row)
            if row_configuration not in selected_configurations:
                continue
            n_bootstraps, possessions, samples = row_configuration
            destination = detailed_destination(
                public_directory,
                profile,
                n_bootstraps,
                possessions,
                samples,
            )
            detailed_count += 1
            index_metric = "support-aware profile recovery"
        elif image_type == "similarity_overview":
            metric = row["metric"]
            if metric not in OVERVIEW_NAMES:
                raise ValueError(f"Unknown overview metric: {metric}")
            destination = overview_destination(public_directory, profile, metric)
            n_bootstraps = ""
            possessions = ""
            samples = ""
            index_metric = metric.replace("_", "-")
            overview_count += 1
        else:
            raise ValueError(f"Unknown plot type in heatmap manifest: {image_type}")

        copy_image(source, destination)
        index_rows.append(
            {
                "image_type": image_type.replace("_", "-"),
                "profile": PROFILE_NAMES[profile],
                "metric": index_metric,
                "bootstrap_count": n_bootstraps,
                "possessions_per_bootstrap": possessions,
                "samples_per_possession": samples,
                "relative_path": destination.relative_to(public_directory).as_posix(),
            }
        )

    expected_detailed = len(selected_configurations) * len(PROFILE_NAMES)
    expected_overviews = len(OVERVIEW_NAMES) * len(PROFILE_NAMES)
    if detailed_count != expected_detailed or overview_count != expected_overviews:
        raise RuntimeError(
            "Incomplete public export: "
            f"detailed={detailed_count}/{expected_detailed}, "
            f"overviews={overview_count}/{expected_overviews}"
        )

    write_heatmap_index(public_directory / "heatmap-index.csv", index_rows)
    ranking_output = public_directory / "rankings"
    ranking_output.mkdir()
    shutil.copy2(overall_source, ranking_output / "overall.csv")
    shutil.copy2(by_profile_source, ranking_output / "by-profile.csv")
    write_collection_readme(collection, public_directory, selected_rows)
    return detailed_count, overview_count


def write_results_readme(collection_counts: list[tuple[Collection, int, int]]) -> None:
    collection_rows = []
    for collection, detailed_count, overview_count in collection_counts:
        collection_rows.append(
            f"| [{collection.simulated_games} simulated games]"
            f"(./{collection.public_name}/) | {collection.possessions_per_game} | "
            f"{detailed_count} | {overview_count} |"
        )

    readme = f"""# QUT profile-recovery results

These are the curated, audience-facing outputs from the profile-simulation
study. Results are separated by simulated-game count, and every numerical
component of a path is labeled with what it counts.

| Study | Possessions per game | Detailed heatmaps | Overview heatmaps |
|---|---:|---:|---:|
{chr(10).join(collection_rows)}

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
"""
    (RESULTS_ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    collection_counts = []
    for collection in COLLECTIONS:
        detailed_count, overview_count = export_collection(collection)
        collection_counts.append((collection, detailed_count, overview_count))
        print(
            f"Exported {collection.public_name}: "
            f"{detailed_count} detailed + {overview_count} overview heatmaps"
        )
    write_results_readme(collection_counts)


if __name__ == "__main__":
    main()
