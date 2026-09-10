#!/usr/bin/env python3
"""Prepare paired profile-recovery inputs for the R thin-plate spline runner.

The exporter recreates the synthetic trajectories and planted outcomes used by
``profile_sims.ipynb`` and materializes the *same* independently seeded
bootstrap slots used by its nested-prefix QUT sweep.  A stream is identified by
analysis scope, possessions per bootstrap, and samples per possession.  Its X
file is shared by all planted profiles; only the four binary outcome files vary.

Raw file layout
---------------
``X.u8`` is uint8 with logical shape ``(B_max, P, 200)`` in C order.  Features
are contiguous within a possession and use ``flat_index = x_bin * 10 + y_bin``.
For one R slot, use ``matrix(as.integer(bytes), nrow=P, ncol=200, byrow=TRUE)``.

Each ``y__<profile>.u8`` is uint8 with shape ``(B_max, P)`` in C order.  The
first B slots are the nested prefix for a configuration with B bootstraps.

Generation is resumable at slot boundaries.  Data are built in a per-stream
partial directory, every committed slot is hashed, and the directory is renamed
atomically only after all files and their final hashes validate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FORMAT_VERSION = "spline_profile_inputs_v1"
COMPARISON_LABEL = "top5_tp_k20"

X_BINS = 20
Y_BINS = 10
N_CELLS = X_BINS * Y_BINS
POSSESSION_KEYS = ("game_id", "possession_number")

SIMULATION_TRAIN_GAMES = (
    21500649,
    21500390,
    21500188,
    21500355,
    21500057,
    21500009,
    21500540,
    21500131,
    21500272,
    21500196,
)
POSSESSIONS_PER_GAME = 92
QUARTERS_PER_GAME = 4
POSSESSIONS_PER_QUARTER = POSSESSIONS_PER_GAME // QUARTERS_PER_GAME
TRAJECTORY_SEED = 142
OUTCOME_SEED = 242
BASELINE_SCORING_PROBABILITY = 0.1
MC_NULL = 100
ALPHA = 0.05
MIN_SOURCE_SAMPLES = 100

SWEEP_DESIGN_VERSION = "factorial_nested_prefix_median_v1"
SEED_POLICY_VERSION = "sha256_slot_seed_v1"
MAP_ESTIMATOR = "nested_prefix_median_v1"
SWEEP_SERIES_ID = "dc0cd9facf"

SCOPES = (
    "full_game",
    "quarter_1",
    "quarter_2",
    "quarter_3",
    "quarter_4",
)
PROFILE_NAMES = (
    "reference",
    "high_y_side",
    "low_y_side",
    "perimeter",
)

# Frozen guardrail for the ranking selected with the adjusted, moderately
# penalized whole-grid cosine metric.  The CSV remains the source of truth, but
# generation stops if its first five rows no longer match this reviewed set.
FROZEN_TOP_FIVE = (
    (1, "b050_p02500_s250", 50, 2500, 250),
    (2, "b050_p01000_s250", 50, 1000, 250),
    (3, "b100_p02500_s250", 100, 2500, 250),
    (4, "b100_p01000_s200", 100, 1000, 200),
    (5, "b100_p01000_s250", 100, 1000, 250),
)

EXPECTED_SOURCE_POSSESSIONS = 758
EXPECTED_SOURCE_ROWS = 283_972
EXPECTED_SOURCE_GAME_FINGERPRINT = "4370615271"
EXPECTED_PROFILE_FINGERPRINT = "2b1671cb4b"


@dataclass(frozen=True)
class RankedConfig:
    performance_rank: int
    config_key: str
    n_bootstraps: int
    possessions_per_bootstrap: int
    samples_per_possession: int


@dataclass(frozen=True)
class StreamSpec:
    scope: str
    possessions_per_bootstrap: int
    samples_per_possession: int
    max_bootstraps: int
    config_keys: tuple[str, ...]

    @property
    def stream_key(self) -> str:
        return (
            f"{self.scope}__p{self.possessions_per_bootstrap:05d}_"
            f"s{self.samples_per_possession:03d}"
        )

    @property
    def x_shape(self) -> tuple[int, int, int]:
        return (self.max_bootstraps, self.possessions_per_bootstrap, N_CELLS)

    @property
    def y_shape(self) -> tuple[int, int]:
        return (self.max_bootstraps, self.possessions_per_bootstrap)

    @property
    def x_nbytes(self) -> int:
        return int(np.prod(self.x_shape, dtype=np.int64))

    @property
    def y_nbytes(self) -> int:
        return int(np.prod(self.y_shape, dtype=np.int64))

    @property
    def x_slot_nbytes(self) -> int:
        return self.possessions_per_bootstrap * N_CELLS

    @property
    def y_slot_nbytes(self) -> int:
        return self.possessions_per_bootstrap


@dataclass
class SimulationData:
    common_qut_input: pd.DataFrame
    profile_outcomes: dict[str, np.ndarray]
    profile_probabilities: dict[str, np.ndarray]
    simulation_tag: str
    summary: dict[str, Any]


@dataclass
class ScopeSource:
    frame: pd.DataFrame
    sector_indices: np.ndarray
    eligible_indices: list[np.ndarray]
    eligible_possession_ids: np.ndarray
    eligible_outcomes: dict[str, np.ndarray]


@dataclass
class SlotData:
    X: np.ndarray
    y: dict[str, np.ndarray]
    selected_possession_ids: np.ndarray


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file_slice(path: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError(
                    f"Unexpected EOF while hashing {path} at offset {offset}."
                )
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def csv_bytes(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def write_or_validate_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"Existing static artifact does not match this run: {path}. "
                "Use a different output root rather than overwriting it."
            )
        return
    atomic_write_bytes(path, payload)


def relative_display(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def make_profiles() -> dict[str, dict[str, Any]]:
    reference = np.zeros((X_BINS, Y_BINS), dtype=float)
    reference[10:18, 0:3] = 0.05
    reference[18:20, 0:3] = 0.05
    reference[18:20, 6] = 0.05
    reference[10:20, 6:10] = 0.05
    reference[18:20, 2:6] = 0.05
    reference[10:18, 2:6] = 0
    reference[10:12, 6:8] = 0
    reference[10:13, :] = 0
    reference[18:20, 2:4] = 0.05

    high_y_side = np.zeros((X_BINS, Y_BINS), dtype=float)
    high_y_side[13:20, 5:10] = 0.05

    low_y_side = np.zeros((X_BINS, Y_BINS), dtype=float)
    low_y_side[13:20, 0:5] = 0.05

    perimeter = np.zeros((X_BINS, Y_BINS), dtype=float)
    perimeter[15:20, 9] = 0.05
    perimeter[14:18, 8:10] = 0.05
    perimeter[14:16, 7:9] = 0.05
    perimeter[13:15, 3:7] = 0.05
    perimeter[14:20, 0] = 0.05
    perimeter[14:18, 1] = 0.05
    perimeter[14:16, 2] = 0.05
    perimeter[13, 0:10] = 0.05
    perimeter[15:20, 3:7] = -0.01
    perimeter[18:20, 3:7] = -0.01
    perimeter[16:18, 3:7] = -0.01
    perimeter[15, 3:7] = 0

    profiles = {
        "reference": {
            "label": "Reference court pattern",
            "movement_effects": reference,
            "end_effects": np.zeros_like(reference),
            "seed": 42,
        },
        "high_y_side": {
            "label": "High-y-side emphasis",
            "movement_effects": high_y_side,
            "end_effects": np.zeros_like(high_y_side),
            "seed": 43,
        },
        "low_y_side": {
            "label": "Low-y-side emphasis",
            "movement_effects": low_y_side,
            "end_effects": np.zeros_like(low_y_side),
            "seed": 44,
        },
        "perimeter": {
            "label": "Stepped perimeter pattern",
            "movement_effects": perimeter,
            "end_effects": np.zeros_like(perimeter),
            "seed": 45,
        },
    }
    if tuple(profiles) != PROFILE_NAMES:
        raise AssertionError("Profile order changed unexpectedly.")
    return profiles


def profile_fingerprint(profiles: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for profile_name in sorted(profiles):
        digest.update(profile_name.encode("utf-8"))
        for effect_name in ("movement_effects", "end_effects"):
            values = np.ascontiguousarray(
                profiles[profile_name][effect_name], dtype="<f8"
            )
            digest.update(effect_name.encode("utf-8"))
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes())
    value = digest.hexdigest()[:10]
    if value != EXPECTED_PROFILE_FINGERPRINT:
        raise AssertionError(
            f"Profile fingerprint changed: expected {EXPECTED_PROFILE_FINGERPRINT}, "
            f"found {value}."
        )
    return value


def source_game_fingerprint() -> str:
    value = hashlib.sha256(
        ",".join(map(str, SIMULATION_TRAIN_GAMES)).encode("utf-8")
    ).hexdigest()[:10]
    if value != EXPECTED_SOURCE_GAME_FINGERPRINT:
        raise AssertionError("The frozen source-game fingerprint changed.")
    return value


def simulation_tag(n_simulated_games: int, profiles: Mapping[str, Any]) -> str:
    return (
        f"games{n_simulated_games}_possessions{POSSESSIONS_PER_GAME}_"
        f"source{source_game_fingerprint()}_grid{X_BINS}x{Y_BINS}_"
        f"trajectory{TRAJECTORY_SEED}_outcome{OUTCOME_SEED}_"
        f"baseline{BASELINE_SCORING_PROBABILITY:.2f}_"
        f"mc{MC_NULL}_alpha{ALPHA:.2f}_min{MIN_SOURCE_SAMPLES}_"
        f"profiles{profile_fingerprint(profiles)}"
    )


def load_ranked_configs(ranking_path: Path) -> tuple[RankedConfig, ...]:
    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)
    ranking = pd.read_csv(ranking_path)
    required = {
        "performance_rank",
        "config_key",
        "n_bootstraps",
        "possessions_per_bootstrap",
        "samples_per_possession",
        "mean_moderately_penalized_whole_grid_cosine",
        "complete_coverage",
    }
    missing = required.difference(ranking.columns)
    if missing:
        raise RuntimeError(
            f"The adjusted 10-game ranking is missing columns: {sorted(missing)}"
        )
    ranking = ranking.copy()
    ranking["performance_rank"] = pd.to_numeric(
        ranking["performance_rank"], errors="raise"
    ).astype(int)
    ranking = ranking.sort_values("performance_rank", kind="stable")
    if ranking["performance_rank"].duplicated().any():
        raise RuntimeError("performance_rank values must be unique.")
    top = ranking.iloc[:5]
    coverage = top["complete_coverage"]
    if coverage.dtype == bool:
        coverage_ok = coverage
    else:
        coverage_ok = coverage.astype(str).str.lower().isin({"true", "1"})
    if not coverage_ok.all():
        raise RuntimeError("A selected top-five configuration has incomplete coverage.")

    configs = tuple(
        RankedConfig(
            performance_rank=int(row.performance_rank),
            config_key=str(row.config_key),
            n_bootstraps=int(row.n_bootstraps),
            possessions_per_bootstrap=int(row.possessions_per_bootstrap),
            samples_per_possession=int(row.samples_per_possession),
        )
        for row in top.itertuples(index=False)
    )
    actual = tuple(
        (
            config.performance_rank,
            config.config_key,
            config.n_bootstraps,
            config.possessions_per_bootstrap,
            config.samples_per_possession,
        )
        for config in configs
    )
    if actual != FROZEN_TOP_FIVE:
        raise RuntimeError(
            "The dynamically selected top five no longer equal the reviewed frozen "
            f"set.\nExpected: {FROZEN_TOP_FIVE}\nActual:   {actual}"
        )
    for config in configs:
        expected_key = (
            f"b{config.n_bootstraps:03d}_"
            f"p{config.possessions_per_bootstrap:05d}_"
            f"s{config.samples_per_possession:03d}"
        )
        if config.config_key != expected_key:
            raise RuntimeError(
                f"Malformed config_key {config.config_key}; expected {expected_key}."
            )
    return configs


def load_qut_metrics_top_five(
    metrics_path: Path,
    configs: Sequence[RankedConfig],
    sim_tag: str,
) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    required = {
        "run_id",
        "simulation_tag",
        "profile",
        "scope",
        "config_key",
        "map_status",
        "whole_grid_cosine",
        "lightly_penalized_whole_grid_cosine",
        "moderately_penalized_whole_grid_cosine",
        "support_cosine",
        "outside_energy_fraction",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise RuntimeError(
            f"QUT representative-map metrics are missing columns: {sorted(missing)}"
        )
    selected_keys = [config.config_key for config in configs]
    filtered = metrics[
        (metrics["simulation_tag"] == sim_tag)
        & metrics["config_key"].isin(selected_keys)
        & metrics["profile"].isin(PROFILE_NAMES)
        & metrics["scope"].isin(SCOPES)
        & (metrics["map_status"] == "ok")
    ].copy()
    expected_combinations = {
        (config_key, profile_name, scope)
        for config_key in selected_keys
        for profile_name in PROFILE_NAMES
        for scope in SCOPES
    }
    actual_combinations = set(
        filtered[["config_key", "profile", "scope"]].itertuples(
            index=False, name=None
        )
    )
    if len(filtered) != 100 or actual_combinations != expected_combinations:
        raise RuntimeError(
            "Expected exactly one successful QUT metric row for each of the "
            "5 configs x 4 profiles x 5 scopes (100 rows); found "
            f"{len(filtered)} rows/{len(actual_combinations)} combinations."
        )
    if not filtered["run_id"].is_unique:
        raise RuntimeError("Filtered QUT metric run_id values are not unique.")
    config_order = {key: index for index, key in enumerate(selected_keys)}
    profile_order = {name: index for index, name in enumerate(PROFILE_NAMES)}
    scope_order = {name: index for index, name in enumerate(SCOPES)}
    filtered["_config_order"] = filtered["config_key"].map(config_order)
    filtered["_profile_order"] = filtered["profile"].map(profile_order)
    filtered["_scope_order"] = filtered["scope"].map(scope_order)
    filtered = (
        filtered.sort_values(
            ["_config_order", "_profile_order", "_scope_order"], kind="stable"
        )
        .drop(columns=["_config_order", "_profile_order", "_scope_order"])
        .reset_index(drop=True)
    )
    if set(filtered["simulation_tag"]) != {sim_tag}:
        raise AssertionError("Filtered QUT metrics have an incorrect simulation tag.")
    return filtered


def build_stream_specs(configs: Sequence[RankedConfig]) -> tuple[StreamSpec, ...]:
    specs: list[StreamSpec] = []
    parameter_pairs = sorted(
        {
            (config.possessions_per_bootstrap, config.samples_per_possession)
            for config in configs
        }
    )
    for scope in SCOPES:
        for possessions, samples in parameter_pairs:
            matching = tuple(
                config
                for config in configs
                if config.possessions_per_bootstrap == possessions
                and config.samples_per_possession == samples
            )
            specs.append(
                StreamSpec(
                    scope=scope,
                    possessions_per_bootstrap=possessions,
                    samples_per_possession=samples,
                    max_bootstraps=max(c.n_bootstraps for c in matching),
                    config_keys=tuple(c.config_key for c in matching),
                )
            )
    if len(specs) != 15:
        raise AssertionError(f"Expected 15 unique scope/P/S streams, found {len(specs)}.")
    if any(spec.max_bootstraps != 100 for spec in specs):
        raise AssertionError("The frozen top-five stream maxima are expected to be 100.")
    return tuple(specs)


def stable_uint32_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def bootstrap_slot_seed(
    sim_tag: str,
    scope: str,
    possessions: int,
    samples: int,
    bootstrap_index: int,
) -> int:
    return stable_uint32_seed(
        sim_tag,
        SWEEP_DESIGN_VERSION,
        SEED_POLICY_VERSION,
        "bootstrap",
        scope,
        int(possessions),
        int(samples),
        int(bootstrap_index),
    )


def load_project_functions(project_root: Path) -> tuple[Any, Any, Any]:
    project_text = str(project_root.resolve())
    if project_text not in sys.path:
        sys.path.insert(0, project_text)
    from funcs import bin_and_flip, grid, iter_bootstrapped_datasets

    return bin_and_flip, grid, iter_bootstrapped_datasets


def sample_complete_possessions(
    source_df: pd.DataFrame,
    n_possessions: int,
) -> tuple[pd.DataFrame, bool, int]:
    source = source_df[source_df["game_id"].isin(SIMULATION_TRAIN_GAMES)].copy()
    keys = source[list(POSSESSION_KEYS)].drop_duplicates()
    if keys.empty:
        raise RuntimeError("No source possessions were found for the selected games.")
    if len(source) != EXPECTED_SOURCE_ROWS or len(keys) != EXPECTED_SOURCE_POSSESSIONS:
        raise RuntimeError(
            "The profile-recovery source cohort changed: expected "
            f"{EXPECTED_SOURCE_ROWS:,} rows/{EXPECTED_SOURCE_POSSESSIONS} possessions, "
            f"found {len(source):,} rows/{len(keys)} possessions."
        )
    replace = n_possessions > len(keys)
    draws = keys.sample(
        n=n_possessions,
        replace=replace,
        random_state=TRAJECTORY_SEED,
    ).reset_index(drop=True)
    draws["_draw_id"] = np.arange(n_possessions, dtype=int)
    sampled = draws.merge(
        source,
        on=list(POSSESSION_KEYS),
        how="left",
        validate="many_to_many",
    )
    sampled["source_possession_number_for_sim"] = sampled["possession_number"]
    sampled["possession_number"] = sampled["_draw_id"]
    return sampled.drop(columns="_draw_id"), replace, len(keys)


def build_simulation(
    project_root: Path,
    source_path: Path,
    n_simulated_games: int,
    profiles: Mapping[str, Mapping[str, Any]],
) -> SimulationData:
    required_source_columns = {
        "game_id",
        "possession_number",
        "period",
        "is_home",
        "x",
        "y",
    }
    source_tracking = pd.read_csv(source_path)
    missing = required_source_columns.difference(source_tracking.columns)
    if missing:
        raise RuntimeError(f"Source CSV is missing columns: {sorted(missing)}")
    missing_games = set(SIMULATION_TRAIN_GAMES).difference(
        source_tracking["game_id"].unique()
    )
    if missing_games:
        raise RuntimeError(f"Source CSV is missing games: {sorted(missing_games)}")

    total_possessions = n_simulated_games * POSSESSIONS_PER_GAME
    sampled, sampled_with_replacement, source_pool_size = sample_complete_possessions(
        source_tracking,
        total_possessions,
    )
    bin_and_flip, _, _ = load_project_functions(project_root)
    binned = bin_and_flip(sampled, flip="no", x_bins=X_BINS, y_bins=Y_BINS)
    binned["sector"] = list(zip(binned["x_bin"], binned["y_bin"]))
    binned["simulation_game"] = (
        binned["possession_number"] // POSSESSIONS_PER_GAME + 1
    )
    binned["possession_in_game"] = (
        binned["possession_number"] % POSSESSIONS_PER_GAME
    )
    binned["quarter"] = (
        binned["possession_in_game"] // POSSESSIONS_PER_QUARTER + 1
    )

    orientation = (
        binned.groupby("game_id", observed=False)["x_bin"]
        .agg(mean_x_bin="mean", n_rows="size")
        .reset_index()
    )
    wrong_way = orientation[orientation["mean_x_bin"] < (X_BINS - 1) / 2]
    if not wrong_way.empty:
        raise RuntimeError(
            "Left-to-right orientation check failed for source games: "
            f"{wrong_way['game_id'].tolist()}"
        )
    game_counts = binned.groupby("simulation_game")["possession_number"].nunique()
    quarter_counts = binned.groupby(["simulation_game", "quarter"])[
        "possession_number"
    ].nunique()
    if not game_counts.eq(POSSESSIONS_PER_GAME).all():
        raise RuntimeError("Synthetic games have incorrect possession counts.")
    if not quarter_counts.eq(POSSESSIONS_PER_QUARTER).all():
        raise RuntimeError("Synthetic quarters have incorrect possession counts.")

    common_columns = [
        "simulation_game",
        "quarter",
        "game_id",
        "source_possession_number_for_sim",
        "possession_number",
        "x_bin",
        "y_bin",
    ]
    common_qut_input = binned[common_columns].copy()
    expected_ids = np.arange(total_possessions, dtype=int)
    actual_ids = np.sort(common_qut_input["possession_number"].unique())
    if not np.array_equal(actual_ids, expected_ids):
        raise RuntimeError("Synthetic possession IDs are not contiguous from zero.")

    profile_outcomes: dict[str, np.ndarray] = {}
    profile_probabilities: dict[str, np.ndarray] = {}
    scoring_summary: dict[str, dict[str, float]] = {}
    for profile_name in PROFILE_NAMES:
        movement_effects = profiles[profile_name]["movement_effects"]
        end_effects = profiles[profile_name]["end_effects"]

        unique_visits = binned.drop_duplicates(
            [*POSSESSION_KEYS, "sector"]
        ).copy()
        unique_visits["movement_adjustment"] = [
            movement_effects[int(x_bin), int(y_bin)]
            for x_bin, y_bin in unique_visits["sector"]
        ]
        movement_adjustments = (
            unique_visits.groupby(list(POSSESSION_KEYS), observed=False)[
                "movement_adjustment"
            ]
            .sum()
            .reset_index()
        )
        end_sectors = binned.drop_duplicates(
            list(POSSESSION_KEYS), keep="last"
        )[[*POSSESSION_KEYS, "sector"]].copy()
        end_sectors["end_adjustment"] = [
            end_effects[int(x_bin), int(y_bin)]
            for x_bin, y_bin in end_sectors["sector"]
        ]
        probabilities = movement_adjustments.merge(
            end_sectors[[*POSSESSION_KEYS, "end_adjustment"]],
            on=list(POSSESSION_KEYS),
            how="left",
            validate="one_to_one",
        )
        probabilities["baseline_probability"] = BASELINE_SCORING_PROBABILITY
        probabilities["scoring_probability"] = (
            probabilities["baseline_probability"]
            + probabilities["movement_adjustment"]
            + probabilities["end_adjustment"]
        ).clip(0, 1)
        rng = np.random.default_rng(OUTCOME_SEED)
        probabilities["outcome_uniform"] = rng.random(len(probabilities))
        probabilities["scored"] = (
            probabilities["outcome_uniform"]
            < probabilities["scoring_probability"]
        ).astype(np.uint8)

        by_id = probabilities.set_index("possession_number")
        if not np.array_equal(np.sort(by_id.index.to_numpy()), expected_ids):
            raise RuntimeError(f"Incomplete outcome IDs for profile {profile_name}.")
        outcomes = by_id.loc[expected_ids, "scored"].to_numpy(dtype=np.uint8)
        probabilities_by_id = by_id.loc[
            expected_ids, "scoring_probability"
        ].to_numpy(dtype=float)
        profile_outcomes[profile_name] = outcomes
        profile_probabilities[profile_name] = probabilities_by_id
        scoring_summary[profile_name] = {
            "mean_scoring_probability": float(probabilities_by_id.mean()),
            "realized_scoring_rate": float(outcomes.mean()),
        }

    unique_source_trajectories = common_qut_input.drop_duplicates(
        ["game_id", "source_possession_number_for_sim"]
    ).shape[0]
    sim_tag = simulation_tag(n_simulated_games, profiles)
    summary = {
        "source_rows": int(len(source_tracking)),
        "source_pool_possessions": int(source_pool_size),
        "synthetic_tracking_rows": int(len(common_qut_input)),
        "synthetic_possessions": int(total_possessions),
        "sampled_with_replacement": bool(sampled_with_replacement),
        "unique_source_trajectories": int(unique_source_trajectories),
        "source_reuse_rate": float(
            1 - unique_source_trajectories / total_possessions
        ),
        "orientation": [
            {
                "game_id": int(row.game_id),
                "mean_x_bin": float(row.mean_x_bin),
                "n_rows": int(row.n_rows),
            }
            for row in orientation.itertuples(index=False)
        ],
        "profiles": scoring_summary,
    }
    return SimulationData(
        common_qut_input=common_qut_input,
        profile_outcomes=profile_outcomes,
        profile_probabilities=profile_probabilities,
        simulation_tag=sim_tag,
        summary=summary,
    )


def build_scope_source(simulation: SimulationData, scope: str) -> ScopeSource:
    if scope == "full_game":
        frame = simulation.common_qut_input.copy()
    else:
        quarter = int(scope.rsplit("_", 1)[1])
        frame = simulation.common_qut_input[
            simulation.common_qut_input["quarter"] == quarter
        ].copy()
    frame.reset_index(drop=True, inplace=True)
    x_bin = frame["x_bin"].to_numpy(dtype=np.int16)
    y_bin = frame["y_bin"].to_numpy(dtype=np.int16)
    if (
        (x_bin < 0).any()
        or (x_bin >= X_BINS).any()
        or (y_bin < 0).any()
        or (y_bin >= Y_BINS).any()
    ):
        raise RuntimeError(f"Out-of-range grid bins in scope {scope}.")
    sector_indices = x_bin * Y_BINS + y_bin

    grouped = frame.groupby(
        "possession_number", sort=False, observed=False
    ).indices
    eligible = [
        (possession_id, np.asarray(indices, dtype=np.intp))
        for possession_id, indices in grouped.items()
        if len(indices) >= MIN_SOURCE_SAMPLES
    ]
    if not eligible:
        raise RuntimeError(f"No eligible possessions in scope {scope}.")
    eligible_possession_ids = np.asarray(
        [int(possession_id) for possession_id, _ in eligible], dtype=np.int64
    )
    eligible_indices = [indices for _, indices in eligible]
    eligible_outcomes = {
        profile_name: simulation.profile_outcomes[profile_name][
            eligible_possession_ids
        ]
        for profile_name in PROFILE_NAMES
    }
    return ScopeSource(
        frame=frame,
        sector_indices=np.asarray(sector_indices, dtype=np.int16),
        eligible_indices=eligible_indices,
        eligible_possession_ids=eligible_possession_ids,
        eligible_outcomes=eligible_outcomes,
    )


def generate_slot(
    source: ScopeSource,
    possessions: int,
    samples: int,
    seed: int,
) -> SlotData:
    rng = np.random.default_rng(seed)
    selected_groups = rng.integers(len(source.eligible_indices), size=possessions)
    X = np.zeros((possessions, N_CELLS), dtype=np.uint8)
    for draw_index, group_index in enumerate(selected_groups):
        sampled_positions = rng.choice(
            source.eligible_indices[int(group_index)],
            size=samples,
            replace=True,
        )
        X[draw_index, np.unique(source.sector_indices[sampled_positions])] = 1
    y = {
        profile_name: np.ascontiguousarray(
            source.eligible_outcomes[profile_name][selected_groups],
            dtype=np.uint8,
        )
        for profile_name in PROFILE_NAMES
    }
    return SlotData(
        X=np.ascontiguousarray(X),
        y=y,
        selected_possession_ids=np.ascontiguousarray(
            source.eligible_possession_ids[selected_groups]
        ),
    )


def check_first_slot_against_funcs(
    project_root: Path,
    simulation: SimulationData,
    source: ScopeSource,
    spec: StreamSpec,
    direct: SlotData,
    seed: int,
) -> dict[str, Any]:
    _, grid, iter_bootstrapped_datasets = load_project_functions(project_root)
    canonical_source = source.frame.copy()
    canonical_source["synthetic_possession_id"] = canonical_source[
        "possession_number"
    ].to_numpy()
    canonical_source["scored"] = simulation.profile_outcomes["reference"][
        canonical_source["possession_number"].to_numpy(dtype=np.int64)
    ]
    canonical_bootstrap = next(
        iter_bootstrapped_datasets(
            canonical_source,
            n_bootstraps=1,
            n_possessions_per_bootstrap=spec.possessions_per_bootstrap,
            samples_per_possession=spec.samples_per_possession,
            min_samples_required=MIN_SOURCE_SAMPLES,
            seed=seed,
            show_progress=False,
        )
    )
    canonical_X, canonical_y, _ = grid(canonical_bootstrap)
    canonical_X = np.asarray(canonical_X.toarray(), dtype=np.uint8)
    canonical_y = np.asarray(canonical_y, dtype=np.uint8)
    canonical_ids = (
        canonical_bootstrap.groupby(
            "possession_number", sort=False, observed=False
        )["synthetic_possession_id"]
        .first()
        .to_numpy(dtype=np.int64)
    )

    failures: list[str] = []
    if not np.array_equal(direct.X, canonical_X):
        failures.append("X")
    if not np.array_equal(direct.y["reference"], canonical_y):
        failures.append("reference y")
    if not np.array_equal(direct.selected_possession_ids, canonical_ids):
        failures.append("selected possession IDs")
    for profile_name in PROFILE_NAMES:
        expected_y = simulation.profile_outcomes[profile_name][canonical_ids]
        if not np.array_equal(direct.y[profile_name], expected_y):
            failures.append(f"{profile_name} y")
    if failures:
        raise AssertionError(
            f"First-slot equality check failed for {spec.stream_key}: "
            + ", ".join(failures)
        )
    x_payload = direct.X.tobytes(order="C")
    return {
        "passed": True,
        "reference_implementation": (
            "funcs.iter_bootstrapped_datasets + funcs.grid"
        ),
        "slot_index": 0,
        "bootstrap_seed": int(seed),
        "x_sha256": sha256_bytes(x_payload),
        "selected_possession_ids_sha256": sha256_bytes(
            direct.selected_possession_ids.astype("<i8", copy=False).tobytes()
        ),
        "y_sha256": {
            profile_name: sha256_bytes(direct.y[profile_name].tobytes())
            for profile_name in PROFILE_NAMES
        },
    }


def stream_identity(
    spec: StreamSpec,
    simulation: SimulationData,
    source_sha256: str,
    ranking_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "simulation_tag": simulation.simulation_tag,
        "source_sha256": source_sha256,
        "ranking_sha256": ranking_sha256,
        "sweep_design_version": SWEEP_DESIGN_VERSION,
        "seed_policy_version": SEED_POLICY_VERSION,
        "stream_key": spec.stream_key,
        "scope": spec.scope,
        "possessions_per_bootstrap": spec.possessions_per_bootstrap,
        "samples_per_possession": spec.samples_per_possession,
        "max_bootstraps": spec.max_bootstraps,
        "n_cells": N_CELLS,
        "x_shape": list(spec.x_shape),
        "y_shape": list(spec.y_shape),
        "x_order": "C: slot, possession, flat_index",
        "flat_index": "x_bin * 10 + y_bin",
        "config_keys": list(spec.config_keys),
        "profiles": list(PROFILE_NAMES),
    }


def raw_paths(directory: Path) -> tuple[Path, dict[str, Path]]:
    return (
        directory / "X.u8",
        {
            profile_name: directory / f"y__{profile_name}.u8"
            for profile_name in PROFILE_NAMES
        },
    )


def initial_resume(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "completed_slots": 0,
        "first_slot_equality": None,
        "slot_hashes": [],
    }


def validate_resume(
    partial_dir: Path,
    spec: StreamSpec,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    resume_path = partial_dir / "resume.json"
    if not resume_path.exists():
        if any(partial_dir.iterdir()):
            raise RuntimeError(f"Partial stream lacks resume.json: {partial_dir}")
        resume = initial_resume(expected_identity)
        atomic_write_bytes(resume_path, json_bytes(resume))
    else:
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
    if resume.get("identity") != dict(expected_identity):
        raise RuntimeError(f"Partial stream identity mismatch: {partial_dir}")
    completed = int(resume.get("completed_slots", -1))
    hashes = resume.get("slot_hashes")
    if completed < 0 or completed > spec.max_bootstraps:
        raise RuntimeError(f"Invalid completed slot count in {resume_path}.")
    if not isinstance(hashes, list) or len(hashes) != completed:
        raise RuntimeError(f"Invalid slot hash ledger in {resume_path}.")
    if completed and not (resume.get("first_slot_equality") or {}).get("passed"):
        raise RuntimeError(f"Missing first-slot equality proof in {resume_path}.")

    x_path, y_paths = raw_paths(partial_dir)
    expected_sizes = {x_path: spec.x_nbytes}
    expected_sizes.update({path: spec.y_nbytes for path in y_paths.values()})
    for path, expected_size in expected_sizes.items():
        if not path.exists():
            if completed:
                raise RuntimeError(f"Missing partial raw file: {path}")
            with path.open("wb") as handle:
                handle.truncate(expected_size)
                handle.flush()
                os.fsync(handle.fileno())
        elif path.stat().st_size != expected_size:
            if completed:
                raise RuntimeError(
                    f"Partial file size mismatch for committed data: {path}"
                )
            with path.open("wb") as handle:
                handle.truncate(expected_size)
                handle.flush()
                os.fsync(handle.fileno())

    for slot_index, record in enumerate(hashes):
        expected_seed = bootstrap_slot_seed(
            str(expected_identity["simulation_tag"]),
            spec.scope,
            spec.possessions_per_bootstrap,
            spec.samples_per_possession,
            slot_index,
        )
        if int(record.get("slot_index", -1)) != slot_index:
            raise RuntimeError(f"Noncontiguous slot ledger in {resume_path}.")
        if int(record.get("bootstrap_seed", -1)) != expected_seed:
            raise RuntimeError(f"Slot seed mismatch in {resume_path}.")
        actual_x = sha256_file_slice(
            x_path,
            slot_index * spec.x_slot_nbytes,
            spec.x_slot_nbytes,
        )
        if actual_x != record.get("x_sha256"):
            raise RuntimeError(f"Committed X slot {slot_index} is corrupt: {x_path}")
        y_hashes = record.get("y_sha256", {})
        for profile_name, y_path in y_paths.items():
            actual_y = sha256_file_slice(
                y_path,
                slot_index * spec.y_slot_nbytes,
                spec.y_slot_nbytes,
            )
            if actual_y != y_hashes.get(profile_name):
                raise RuntimeError(
                    f"Committed {profile_name} y slot {slot_index} is corrupt: {y_path}"
                )
    return resume


def complete_stream_state(
    stream_dir: Path,
    spec: StreamSpec,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_path = stream_dir / "stream.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Completed stream lacks stream.json: {stream_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError(f"Stream is not marked complete: {stream_dir}")
    if metadata.get("identity") != dict(expected_identity):
        raise RuntimeError(f"Completed stream identity mismatch: {stream_dir}")
    if not (metadata.get("first_slot_equality") or {}).get("passed"):
        raise RuntimeError(f"Completed stream lacks equality proof: {stream_dir}")

    x_path, y_paths = raw_paths(stream_dir)
    files = metadata.get("files", {})
    x_info = files.get("X", {})
    if x_path.stat().st_size != spec.x_nbytes:
        raise RuntimeError(f"X size mismatch: {x_path}")
    if int(x_info.get("nbytes", -1)) != spec.x_nbytes:
        raise RuntimeError(f"X metadata size mismatch: {metadata_path}")
    actual_x_sha = sha256_file(x_path)
    if actual_x_sha != x_info.get("sha256"):
        raise RuntimeError(f"X checksum mismatch: {x_path}")
    y_info = files.get("y", {})
    y_checksums: dict[str, str] = {}
    for profile_name, y_path in y_paths.items():
        info = y_info.get(profile_name, {})
        if y_path.stat().st_size != spec.y_nbytes:
            raise RuntimeError(f"y size mismatch: {y_path}")
        if int(info.get("nbytes", -1)) != spec.y_nbytes:
            raise RuntimeError(f"y metadata size mismatch: {metadata_path}")
        actual_sha = sha256_file(y_path)
        if actual_sha != info.get("sha256"):
            raise RuntimeError(f"y checksum mismatch: {y_path}")
        y_checksums[profile_name] = actual_sha
    return {
        "status": "valid",
        "completed_slots": spec.max_bootstraps,
        "x_sha256": actual_x_sha,
        "y_sha256": y_checksums,
        "first_slot_equality_passed": True,
    }


def inspect_stream(
    streams_root: Path,
    spec: StreamSpec,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    stream_dir = streams_root / spec.stream_key
    partial_dir = streams_root / f".{spec.stream_key}.partial"
    if stream_dir.exists() and partial_dir.exists():
        raise RuntimeError(
            f"Both completed and partial directories exist for {spec.stream_key}."
        )
    if stream_dir.exists():
        return complete_stream_state(stream_dir, spec, identity)
    if partial_dir.exists():
        resume = validate_resume(partial_dir, spec, identity)
        return {
            "status": "partial",
            "completed_slots": int(resume["completed_slots"]),
            "x_sha256": "",
            "y_sha256": {profile_name: "" for profile_name in PROFILE_NAMES},
            "first_slot_equality_passed": bool(
                (resume.get("first_slot_equality") or {}).get("passed", False)
            ),
        }
    return {
        "status": "pending",
        "completed_slots": 0,
        "x_sha256": "",
        "y_sha256": {profile_name: "" for profile_name in PROFILE_NAMES},
        "first_slot_equality_passed": False,
    }


def write_slot(
    x_handle: Any,
    y_handles: Mapping[str, Any],
    spec: StreamSpec,
    slot_index: int,
    slot: SlotData,
) -> dict[str, Any]:
    x_payload = slot.X.tobytes(order="C")
    if len(x_payload) != spec.x_slot_nbytes:
        raise AssertionError("Unexpected X slot byte count.")
    x_handle.seek(slot_index * spec.x_slot_nbytes)
    if x_handle.write(x_payload) != len(x_payload):
        raise OSError("Short write while storing X.")
    x_handle.flush()
    os.fsync(x_handle.fileno())

    y_hashes: dict[str, str] = {}
    for profile_name, handle in y_handles.items():
        payload = slot.y[profile_name].tobytes(order="C")
        if len(payload) != spec.y_slot_nbytes:
            raise AssertionError("Unexpected y slot byte count.")
        handle.seek(slot_index * spec.y_slot_nbytes)
        if handle.write(payload) != len(payload):
            raise OSError(f"Short write while storing {profile_name} y.")
        handle.flush()
        os.fsync(handle.fileno())
        y_hashes[profile_name] = sha256_bytes(payload)
    return {
        "slot_index": int(slot_index),
        "x_sha256": sha256_bytes(x_payload),
        "y_sha256": y_hashes,
    }


def generate_stream(
    project_root: Path,
    streams_root: Path,
    simulation: SimulationData,
    scope_source: ScopeSource,
    spec: StreamSpec,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    stream_dir = streams_root / spec.stream_key
    partial_dir = streams_root / f".{spec.stream_key}.partial"
    if stream_dir.exists():
        return complete_stream_state(stream_dir, spec, identity)
    partial_dir.mkdir(parents=True, exist_ok=True)
    resume = validate_resume(partial_dir, spec, identity)
    x_path, y_paths = raw_paths(partial_dir)

    with x_path.open("r+b") as x_handle:
        y_handles = {
            profile_name: path.open("r+b")
            for profile_name, path in y_paths.items()
        }
        try:
            for slot_index in range(
                int(resume["completed_slots"]), spec.max_bootstraps
            ):
                seed = bootstrap_slot_seed(
                    simulation.simulation_tag,
                    spec.scope,
                    spec.possessions_per_bootstrap,
                    spec.samples_per_possession,
                    slot_index,
                )
                slot = generate_slot(
                    scope_source,
                    spec.possessions_per_bootstrap,
                    spec.samples_per_possession,
                    seed,
                )
                if slot_index == 0:
                    resume["first_slot_equality"] = check_first_slot_against_funcs(
                        project_root,
                        simulation,
                        scope_source,
                        spec,
                        slot,
                        seed,
                    )
                    print(f"  first-slot equality passed: {spec.stream_key}")
                record = write_slot(
                    x_handle, y_handles, spec, slot_index, slot
                )
                record["bootstrap_seed"] = int(seed)
                resume["slot_hashes"].append(record)
                resume["completed_slots"] = slot_index + 1
                atomic_write_bytes(
                    partial_dir / "resume.json", json_bytes(resume)
                )
                print(
                    f"  {spec.stream_key}: slot {slot_index + 1}/"
                    f"{spec.max_bootstraps}",
                    flush=True,
                )
        finally:
            for handle in y_handles.values():
                handle.close()

    x_sha = sha256_file(x_path)
    y_sha = {
        profile_name: sha256_file(path)
        for profile_name, path in y_paths.items()
    }
    metadata = {
        "status": "complete",
        "identity": dict(identity),
        "first_slot_equality": resume["first_slot_equality"],
        "files": {
            "X": {
                "path": "X.u8",
                "dtype": "uint8",
                "shape": list(spec.x_shape),
                "order": "C",
                "nbytes": spec.x_nbytes,
                "sha256": x_sha,
            },
            "y": {
                profile_name: {
                    "path": f"y__{profile_name}.u8",
                    "dtype": "uint8",
                    "shape": list(spec.y_shape),
                    "order": "C",
                    "nbytes": spec.y_nbytes,
                    "sha256": y_sha[profile_name],
                }
                for profile_name in PROFILE_NAMES
            },
        },
    }
    atomic_write_bytes(partial_dir / "stream.json", json_bytes(metadata))
    if stream_dir.exists():
        raise RuntimeError(f"Refusing to replace existing stream: {stream_dir}")
    os.replace(partial_dir, stream_dir)
    return {
        "status": "valid",
        "completed_slots": spec.max_bootstraps,
        "x_sha256": x_sha,
        "y_sha256": y_sha,
        "first_slot_equality_passed": True,
    }


CONFIG_MANIFEST_FIELDS = (
    "performance_rank",
    "config_key",
    "n_bootstraps",
    "possessions_per_bootstrap",
    "samples_per_possession",
    "scope",
    "stream_key",
    "prefix_slots",
    "x_path",
    *tuple(f"y_{profile_name}_path" for profile_name in PROFILE_NAMES),
)

SLOT_MANIFEST_FIELDS = (
    "stream_key",
    "scope",
    "possessions_per_bootstrap",
    "samples_per_possession",
    "slot_index",
    "bootstrap_seed",
    "x_byte_offset",
    "x_nbytes",
    "y_byte_offset",
    "y_nbytes",
    "seed_policy_version",
)

STREAM_MANIFEST_FIELDS = (
    "stream_key",
    "simulation_tag",
    "simulated_games",
    "scope",
    "possessions_per_bootstrap",
    "samples_per_possession",
    "max_bootstraps",
    "n_cells",
    "config_keys",
    "seed_policy_version",
    "slot_manifest_path",
    "x_dtype",
    "x_layout",
    "column_order",
    "x_path",
    "x_shape",
    "x_nbytes",
    "x_sha256",
    *tuple(
        field
        for profile_name in PROFILE_NAMES
        for field in (
            f"y_{profile_name}_path",
            f"y_{profile_name}_shape",
            f"y_{profile_name}_nbytes",
            f"y_{profile_name}_sha256",
        )
    ),
    "status",
    "completed_slots",
    "first_slot_equality_passed",
)


def config_manifest_rows(
    configs: Sequence[RankedConfig],
) -> list[dict[str, Any]]:
    rows = []
    for config in configs:
        for scope in SCOPES:
            stream_key = (
                f"{scope}__p{config.possessions_per_bootstrap:05d}_"
                f"s{config.samples_per_possession:03d}"
            )
            row: dict[str, Any] = {
                "performance_rank": config.performance_rank,
                "config_key": config.config_key,
                "n_bootstraps": config.n_bootstraps,
                "possessions_per_bootstrap": config.possessions_per_bootstrap,
                "samples_per_possession": config.samples_per_possession,
                "scope": scope,
                "stream_key": stream_key,
                "prefix_slots": config.n_bootstraps,
                "x_path": f"streams/{stream_key}/X.u8",
            }
            row.update(
                {
                    f"y_{profile_name}_path": (
                        f"streams/{stream_key}/y__{profile_name}.u8"
                    )
                    for profile_name in PROFILE_NAMES
                }
            )
            rows.append(row)
    return rows


def slot_manifest_rows(
    specs: Sequence[StreamSpec], sim_tag: str
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        for slot_index in range(spec.max_bootstraps):
            rows.append(
                {
                    "stream_key": spec.stream_key,
                    "scope": spec.scope,
                    "possessions_per_bootstrap": spec.possessions_per_bootstrap,
                    "samples_per_possession": spec.samples_per_possession,
                    "slot_index": slot_index,
                    "bootstrap_seed": bootstrap_slot_seed(
                        sim_tag,
                        spec.scope,
                        spec.possessions_per_bootstrap,
                        spec.samples_per_possession,
                        slot_index,
                    ),
                    "x_byte_offset": slot_index * spec.x_slot_nbytes,
                    "x_nbytes": spec.x_slot_nbytes,
                    "y_byte_offset": slot_index * spec.y_slot_nbytes,
                    "y_nbytes": spec.y_slot_nbytes,
                    "seed_policy_version": SEED_POLICY_VERSION,
                }
            )
    return rows


def stream_manifest_rows(
    specs: Sequence[StreamSpec],
    states: Mapping[str, Mapping[str, Any]],
    sim_tag: str,
    n_simulated_games: int,
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        state = states[spec.stream_key]
        stream_prefix = f"streams/{spec.stream_key}"
        row: dict[str, Any] = {
            "stream_key": spec.stream_key,
            "simulation_tag": sim_tag,
            "simulated_games": n_simulated_games,
            "scope": spec.scope,
            "possessions_per_bootstrap": spec.possessions_per_bootstrap,
            "samples_per_possession": spec.samples_per_possession,
            "max_bootstraps": spec.max_bootstraps,
            "n_cells": N_CELLS,
            "config_keys": ";".join(spec.config_keys),
            "seed_policy_version": SEED_POLICY_VERSION,
            "slot_manifest_path": "slot_manifest.csv",
            "x_dtype": "uint8",
            "x_layout": "C: slot, possession, flat_index",
            "column_order": "x-major/y-fast: flat_index=x_bin*10+y_bin",
            "x_path": f"{stream_prefix}/X.u8",
            "x_shape": "x".join(map(str, spec.x_shape)),
            "x_nbytes": spec.x_nbytes,
            "x_sha256": state["x_sha256"],
            "status": state["status"],
            "completed_slots": state["completed_slots"],
            "first_slot_equality_passed": str(
                bool(state["first_slot_equality_passed"])
            ).lower(),
        }
        for profile_name in PROFILE_NAMES:
            row.update(
                {
                    f"y_{profile_name}_path": (
                        f"{stream_prefix}/y__{profile_name}.u8"
                    ),
                    f"y_{profile_name}_shape": "x".join(
                        map(str, spec.y_shape)
                    ),
                    f"y_{profile_name}_nbytes": spec.y_nbytes,
                    f"y_{profile_name}_sha256": state["y_sha256"][
                        profile_name
                    ],
                }
            )
        rows.append(row)
    return rows


def planted_profile_rows(
    profiles: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for profile_name in PROFILE_NAMES:
        profile = profiles[profile_name]
        for x_bin in range(X_BINS):
            for y_bin in range(Y_BINS):
                rows.append(
                    {
                        "profile": profile_name,
                        "profile_label": profile["label"],
                        "profile_seed": profile["seed"],
                        "x_bin": x_bin,
                        "y_bin": y_bin,
                        "flat_index": x_bin * Y_BINS + y_bin,
                        "movement_effect": float(
                            profile["movement_effects"][x_bin, y_bin]
                        ),
                        "end_effect": float(
                            profile["end_effects"][x_bin, y_bin]
                        ),
                    }
                )
    return rows


PLANTED_PROFILE_FIELDS = (
    "profile",
    "profile_label",
    "profile_seed",
    "x_bin",
    "y_bin",
    "flat_index",
    "movement_effect",
    "end_effect",
)


def global_metadata(
    project_root: Path,
    source_path: Path,
    ranking_path: Path,
    qut_metrics_path: Path,
    source_sha256: str,
    ranking_sha256: str,
    qut_metrics_sha256: str,
    n_simulated_games: int,
    profiles: Mapping[str, Mapping[str, Any]],
    configs: Sequence[RankedConfig],
    specs: Sequence[StreamSpec],
    simulation: SimulationData,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "comparison_label": COMPARISON_LABEL,
        "simulation_tag": simulation.simulation_tag,
        "n_simulated_games": n_simulated_games,
        "possessions_per_game": POSSESSIONS_PER_GAME,
        "quarters_per_game": QUARTERS_PER_GAME,
        "trajectory_seed": TRAJECTORY_SEED,
        "outcome_seed": OUTCOME_SEED,
        "baseline_scoring_probability": BASELINE_SCORING_PROBABILITY,
        "minimum_source_samples": MIN_SOURCE_SAMPLES,
        "source_games": list(SIMULATION_TRAIN_GAMES),
        "source_game_fingerprint": source_game_fingerprint(),
        "profile_definition_fingerprint": profile_fingerprint(profiles),
        "source_csv": relative_display(source_path, project_root),
        "source_sha256": source_sha256,
        "ranking_csv": relative_display(ranking_path, project_root),
        "ranking_sha256": ranking_sha256,
        "qut_metrics_source_csv": relative_display(qut_metrics_path, project_root),
        "qut_metrics_source_sha256": qut_metrics_sha256,
        "qut_metrics_export": "qut_metrics_top5.csv",
        "qut_metrics_export_rows": 100,
        "ranking_selection": "first five rows by performance_rank",
        "ranking_guardrail": [list(row) for row in FROZEN_TOP_FIVE],
        "top_five": [config.__dict__ for config in configs],
        "n_unique_streams": len(specs),
        "scopes": list(SCOPES),
        "profiles": list(PROFILE_NAMES),
        "grid": {
            "x_bins": X_BINS,
            "y_bins": Y_BINS,
            "n_cells": N_CELLS,
            "flat_index": "x_bin * 10 + y_bin",
        },
        "raw_layout": {
            "dtype": "uint8",
            "byte_order": "not applicable (one-byte elements)",
            "X": "C order: bootstrap slot, possession, flat_index",
            "y": "C order: bootstrap slot, possession",
            "r_X_slot_reader": (
                "matrix(as.integer(readBin(..., what='raw', n=P*200)), "
                "nrow=P, ncol=200, byrow=TRUE)"
            ),
        },
        "bootstrap": {
            "sweep_design_version": SWEEP_DESIGN_VERSION,
            "seed_policy_version": SEED_POLICY_VERSION,
            "map_estimator": MAP_ESTIMATOR,
            "X_is_binary_unique_visited_sector": True,
            "profiles_share_X": True,
            "nested_prefix": True,
        },
        "intended_spline": {"basis": "thin_plate", "k": 20},
        "simulation_summary": simulation.summary,
    }


def initialize_static_artifacts(
    output_root: Path,
    metadata: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    configs: Sequence[RankedConfig],
    specs: Sequence[StreamSpec],
    sim_tag: str,
    qut_metrics: pd.DataFrame,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "streams").mkdir(parents=True, exist_ok=True)
    write_or_validate_exact(output_root / "metadata.json", json_bytes(metadata))
    write_or_validate_exact(
        output_root / "planted_profiles.csv",
        csv_bytes(planted_profile_rows(profiles), PLANTED_PROFILE_FIELDS),
    )
    write_or_validate_exact(
        output_root / "config_manifest.csv",
        csv_bytes(config_manifest_rows(configs), CONFIG_MANIFEST_FIELDS),
    )
    write_or_validate_exact(
        output_root / "slot_manifest.csv",
        csv_bytes(slot_manifest_rows(specs, sim_tag), SLOT_MANIFEST_FIELDS),
    )
    write_or_validate_exact(
        output_root / "qut_metrics_top5.csv",
        dataframe_csv_bytes(qut_metrics),
    )


def validate_static_artifacts(
    output_root: Path,
    metadata: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    configs: Sequence[RankedConfig],
    specs: Sequence[StreamSpec],
    sim_tag: str,
    qut_metrics: pd.DataFrame,
) -> None:
    expected = {
        output_root / "metadata.json": json_bytes(metadata),
        output_root / "planted_profiles.csv": csv_bytes(
            planted_profile_rows(profiles), PLANTED_PROFILE_FIELDS
        ),
        output_root / "config_manifest.csv": csv_bytes(
            config_manifest_rows(configs), CONFIG_MANIFEST_FIELDS
        ),
        output_root / "slot_manifest.csv": csv_bytes(
            slot_manifest_rows(specs, sim_tag), SLOT_MANIFEST_FIELDS
        ),
        output_root / "qut_metrics_top5.csv": dataframe_csv_bytes(qut_metrics),
    }
    for path, payload in expected.items():
        if not path.exists() or path.read_bytes() != payload:
            raise RuntimeError(f"Static artifact is absent or incompatible: {path}")


def write_stream_manifest(
    output_root: Path,
    specs: Sequence[StreamSpec],
    states: Mapping[str, Mapping[str, Any]],
    sim_tag: str,
    n_simulated_games: int,
) -> None:
    atomic_write_bytes(
        output_root / "stream_manifest.csv",
        csv_bytes(
            stream_manifest_rows(
                specs, states, sim_tag, n_simulated_games
            ),
            STREAM_MANIFEST_FIELDS,
        ),
    )


def select_specs(
    specs: Sequence[StreamSpec],
    scopes: Sequence[str] | None,
    stream_keys: Sequence[str] | None,
    max_streams: int | None,
) -> tuple[StreamSpec, ...]:
    selected = list(specs)
    if scopes:
        selected = [spec for spec in selected if spec.scope in set(scopes)]
    if stream_keys:
        known = {spec.stream_key for spec in specs}
        unknown = set(stream_keys).difference(known)
        if unknown:
            raise ValueError(f"Unknown stream key(s): {sorted(unknown)}")
        requested = set(stream_keys)
        selected = [spec for spec in selected if spec.stream_key in requested]
    if max_streams is not None:
        if max_streams < 1:
            raise ValueError("--max-streams must be at least 1.")
        selected = selected[:max_streams]
    if not selected:
        raise ValueError("The stream filters selected no work.")
    return tuple(selected)


def resolve_path(project_root: Path, value: str | None, default: Path) -> Path:
    path = default if value is None else Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        help="Project root (defaults to the parent of this script's directory).",
    )
    parser.add_argument(
        "--n-simulated-games",
        type=int,
        choices=(8, 10),
        default=10,
        help="Generate the required 10-game inputs or optional 8-game validation inputs.",
    )
    parser.add_argument("--source-csv", help="Override the tracking source CSV.")
    parser.add_argument(
        "--ranking-path",
        help="Override the adjusted 10-game ranking CSV (selection is always guarded).",
    )
    parser.add_argument(
        "--qut-metrics-path",
        help=(
            "Override the matching representative-map QUT metrics CSV. "
            "The default is derived from the selected study's simulation tag."
        ),
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Output root. Default: profile_sim_results/spline_comparison/"
            "games<N>_top5_tp_k20/inputs."
        ),
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=SCOPES,
        help="Generate/validate only these scopes while retaining the full manifest.",
    )
    parser.add_argument(
        "--stream-keys",
        nargs="+",
        help="Generate/validate only the named scope/P/S streams.",
    )
    parser.add_argument(
        "--max-streams",
        type=int,
        help="Limit selected streams for a pilot; a later unfiltered run resumes the rest.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the guarded configuration/stream plan without reading source data.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Hash and validate existing selected outputs without generating data.",
    )
    parser.add_argument(
        "--check-first-slot",
        metavar="STREAM_KEY",
        help="Run one no-write direct-vs-funcs equality check and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inferred_root = Path(__file__).resolve().parents[1]
    project_root = resolve_path(inferred_root, args.project_root, inferred_root)
    ranking_path = resolve_path(
        project_root,
        args.ranking_path,
        project_root / "profile_sim_results" / "tables" / "10_games" /
        "ranking_overall.csv",
    )
    source_path = resolve_path(
        project_root,
        args.source_csv,
        project_root / "multi_game_data" / "multi_game_data_half_court_filtered.csv",
    )
    output_root = resolve_path(
        project_root,
        args.output_root,
        project_root / "profile_sim_results" / "spline_comparison" /
        f"games{args.n_simulated_games}_{COMPARISON_LABEL}" / "inputs",
    )

    profiles = make_profiles()
    configs = load_ranked_configs(ranking_path)
    specs = build_stream_specs(configs)
    sim_tag = simulation_tag(args.n_simulated_games, profiles)
    qut_metrics_path = resolve_path(
        project_root,
        args.qut_metrics_path,
        project_root / "profile_sim_results" / "nested_maps" /
        f"{sim_tag}_plandc0cd9facf" / "representative_map_similarity.csv",
    )
    selected_specs = select_specs(
        specs,
        args.scopes,
        args.stream_keys,
        args.max_streams,
    )

    if args.plan_only:
        print(
            json.dumps(
                {
                    "simulation_tag": sim_tag,
                    "ranking_path": str(ranking_path),
                    "qut_metrics_path": str(qut_metrics_path),
                    "output_root": str(output_root),
                    "top_five": [config.__dict__ for config in configs],
                    "selected_streams": [spec.stream_key for spec in selected_specs],
                    "all_streams": [spec.stream_key for spec in specs],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_sha256 = sha256_file(source_path)
    ranking_sha256 = sha256_file(ranking_path)
    qut_metrics_sha256 = sha256_file(qut_metrics_path)
    qut_metrics = load_qut_metrics_top_five(
        qut_metrics_path,
        configs,
        sim_tag,
    )
    print(f"Reconstructing {sim_tag}", flush=True)
    simulation = build_simulation(
        project_root,
        source_path,
        args.n_simulated_games,
        profiles,
    )
    scope_sources: dict[str, ScopeSource] = {}

    if args.check_first_slot:
        lookup = {spec.stream_key: spec for spec in specs}
        if args.check_first_slot not in lookup:
            raise ValueError(
                f"Unknown --check-first-slot value: {args.check_first_slot}"
            )
        spec = lookup[args.check_first_slot]
        source = build_scope_source(simulation, spec.scope)
        seed = bootstrap_slot_seed(
            simulation.simulation_tag,
            spec.scope,
            spec.possessions_per_bootstrap,
            spec.samples_per_possession,
            0,
        )
        slot = generate_slot(
            source,
            spec.possessions_per_bootstrap,
            spec.samples_per_possession,
            seed,
        )
        result = check_first_slot_against_funcs(
            project_root, simulation, source, spec, slot, seed
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    metadata = global_metadata(
        project_root,
        source_path,
        ranking_path,
        qut_metrics_path,
        source_sha256,
        ranking_sha256,
        qut_metrics_sha256,
        args.n_simulated_games,
        profiles,
        configs,
        specs,
        simulation,
    )
    if args.validate_only:
        if not output_root.exists():
            raise FileNotFoundError(output_root)
        validate_static_artifacts(
            output_root,
            metadata,
            profiles,
            configs,
            specs,
            sim_tag,
            qut_metrics,
        )
    else:
        initialize_static_artifacts(
            output_root,
            metadata,
            profiles,
            configs,
            specs,
            sim_tag,
            qut_metrics,
        )

    streams_root = output_root / "streams"
    identities = {
        spec.stream_key: stream_identity(
            spec, simulation, source_sha256, ranking_sha256
        )
        for spec in specs
    }
    states = {
        spec.stream_key: inspect_stream(
            streams_root, spec, identities[spec.stream_key]
        )
        for spec in specs
    }
    if not args.validate_only:
        write_stream_manifest(
            output_root,
            specs,
            states,
            sim_tag,
            args.n_simulated_games,
        )

    if args.validate_only:
        selected_states = [states[spec.stream_key] for spec in selected_specs]
        for spec in selected_specs:
            state = states[spec.stream_key]
            print(
                f"{spec.stream_key}: {state['status']} "
                f"({state['completed_slots']}/{spec.max_bootstraps})"
            )
        return 0 if all(state["status"] == "valid" for state in selected_states) else 1

    for number, spec in enumerate(selected_specs, start=1):
        state = states[spec.stream_key]
        if state["status"] == "valid":
            print(
                f"SKIP valid stream {number}/{len(selected_specs)}: "
                f"{spec.stream_key}"
            )
            continue
        print(
            f"STREAM {number}/{len(selected_specs)}: {spec.stream_key} "
            f"(resume at {state['completed_slots']}/{spec.max_bootstraps})",
            flush=True,
        )
        if spec.scope not in scope_sources:
            scope_sources[spec.scope] = build_scope_source(
                simulation, spec.scope
            )
        states[spec.stream_key] = generate_stream(
            project_root,
            streams_root,
            simulation,
            scope_sources[spec.scope],
            spec,
            identities[spec.stream_key],
        )
        write_stream_manifest(
            output_root,
            specs,
            states,
            sim_tag,
            args.n_simulated_games,
        )

    valid_count = sum(state["status"] == "valid" for state in states.values())
    partial_count = sum(state["status"] == "partial" for state in states.values())
    print(
        f"Input preparation state: {valid_count}/{len(specs)} valid, "
        f"{partial_count} partial. Manifest: {output_root / 'stream_manifest.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
