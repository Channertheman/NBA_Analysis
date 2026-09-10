"""Render validated QUT-versus-spline profile-recovery comparisons.

This script creates two publication-ready figures from the numeric artifacts:

1. A four-profile spatial plate for one frozen configuration/scope, showing the
   planted profile, the QUT prefix-median map, and the thin-plate spline map.
2. Two paired scatter plots over all 100 frozen config/profile/scope cases,
   comparing the primary moderate score and outside-support energy.

Recovered maps in the spatial plate are adjusted exactly as in the v2 metric
(subtract the median outside the planted support), then every displayed map is
unit-L2 normalized.  That display normalization compares spatial shape rather
than coefficient magnitude; the annotated metrics are calculated from the
unnormalized, background-adjusted maps.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


NX = 20
NY = 10
N_CELLS = NX * NY
PROFILES: tuple[str, ...] = (
    "reference",
    "high_y_side",
    "low_y_side",
    "perimeter",
)
PROFILE_LABELS: Mapping[str, str] = {
    "reference": "Reference court pattern",
    "high_y_side": "High-y-side emphasis",
    "low_y_side": "Low-y-side emphasis",
    "perimeter": "Stepped perimeter pattern",
}
PROFILE_COLORS: Mapping[str, str] = {
    "reference": "#0072B2",
    "high_y_side": "#D55E00",
    "low_y_side": "#009E73",
    "perimeter": "#CC79A7",
}
SCOPES: tuple[str, ...] = (
    "full_game",
    "quarter_1",
    "quarter_2",
    "quarter_3",
    "quarter_4",
)
SCOPE_LABELS: Mapping[str, str] = {
    "full_game": "Full game",
    "quarter_1": "Quarter 1",
    "quarter_2": "Quarter 2",
    "quarter_3": "Quarter 3",
    "quarter_4": "Quarter 4",
}
SCOPE_MARKERS: Mapping[str, str] = {
    "full_game": "o",
    "quarter_1": "s",
    "quarter_2": "^",
    "quarter_3": "D",
    "quarter_4": "P",
}
FROZEN_CONFIGS: Mapping[str, tuple[int, int, int]] = {
    "b050_p02500_s250": (50, 2500, 250),
    "b050_p01000_s250": (50, 1000, 250),
    "b100_p02500_s250": (100, 2500, 250),
    "b100_p01000_s200": (100, 1000, 200),
    "b100_p01000_s250": (100, 1000, 250),
}
DEFAULT_CONFIG = "b050_p02500_s250"
DEFAULT_SCOPE = "full_game"
SIMILARITY_METHOD = "whole_grid_cosine_dual_leakage_penalty_v2"
SUPPORT_TOLERANCE = 1e-12
LIGHT_EXPONENT = 0.5
MODERATE_EXPONENT = 1.0
METRIC_ATOL = 2e-7
EXPECTED_PAIR_ROWS = len(FROZEN_CONFIGS) * len(PROFILES) * len(SCOPES)


class VisualInputError(ValueError):
    """Raised when a source artifact cannot support a trusted figure."""


@dataclass(frozen=True)
class MetricValues:
    whole_grid_cosine: float
    support_cosine: float
    outside_energy_fraction: float
    light_score: float
    moderate_score: float
    background: float
    adjusted: np.ndarray


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise VisualInputError(f"{label} is missing required columns: {missing}")


def _single_text(frame: pd.DataFrame, column: str, label: str) -> str:
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise VisualInputError(f"{label} column {column!r} has missing values")
    unique = values.unique().tolist()
    if len(unique) != 1:
        raise VisualInputError(
            f"{label} column {column!r} must have one value; found {unique}"
        )
    return str(unique[0])


def _validate_pairs(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise VisualInputError(f"Paired comparison CSV does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "simulation_tag",
        "simulated_games",
        "similarity_method",
        "light_leakage_exponent",
        "moderate_leakage_exponent",
        "run_id",
        "profile",
        "scope",
        "config_key",
        "n_bootstraps",
        "possessions_per_bootstrap",
        "samples_per_possession",
        "qut_whole_grid_cosine",
        "spline_whole_grid_cosine",
        "qut_moderately_penalized_whole_grid_cosine",
        "spline_moderately_penalized_whole_grid_cosine",
        "qut_support_cosine",
        "spline_support_cosine",
        "qut_outside_energy_fraction",
        "spline_outside_energy_fraction",
        "moderate_score_winner",
        "winner_tolerance",
    }
    _require_columns(frame, required, "paired comparison CSV")
    if len(frame) != EXPECTED_PAIR_ROWS:
        raise VisualInputError(
            f"Paired comparison CSV must have {EXPECTED_PAIR_ROWS} rows; "
            f"found {len(frame)}"
        )
    if not frame["run_id"].is_unique:
        raise VisualInputError("Paired comparison run_id values are not unique")

    simulation_tag = _single_text(frame, "simulation_tag", "paired comparison")
    method = _single_text(frame, "similarity_method", "paired comparison")
    if method != SIMILARITY_METHOD:
        raise VisualInputError(
            f"Expected similarity method {SIMILARITY_METHOD!r}; found {method!r}"
        )
    games = pd.to_numeric(frame["simulated_games"], errors="raise").to_numpy(float)
    if not np.isfinite(games).all() or not np.all(games == 10):
        raise VisualInputError("Visuals require the complete ten-game comparison")

    expected_identity = {
        (profile, scope, config)
        for profile in PROFILES
        for scope in SCOPES
        for config in FROZEN_CONFIGS
    }
    observed_identity = set(
        frame[["profile", "scope", "config_key"]].itertuples(index=False, name=None)
    )
    if observed_identity != expected_identity:
        raise VisualInputError(
            "Paired comparison does not contain the exact frozen grid"
        )

    expected_run_ids = {
        f"{profile}__{scope}__{config}" for profile, scope, config in expected_identity
    }
    if set(frame["run_id"].astype(str)) != expected_run_ids:
        raise VisualInputError(
            "Paired comparison run_id values do not match their grid"
        )

    for config, settings in FROZEN_CONFIGS.items():
        rows = frame.loc[frame["config_key"].eq(config)]
        for column, expected in zip(
            (
                "n_bootstraps",
                "possessions_per_bootstrap",
                "samples_per_possession",
            ),
            settings,
        ):
            values = pd.to_numeric(rows[column], errors="raise").to_numpy(float)
            if not np.all(values == expected):
                raise VisualInputError(
                    f"Paired comparison settings disagree with {config}: {column}"
                )

    exponents = {
        "light_leakage_exponent": LIGHT_EXPONENT,
        "moderate_leakage_exponent": MODERATE_EXPONENT,
    }
    for column, expected in exponents.items():
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.allclose(
            values, expected, rtol=0.0, atol=1e-12
        ):
            raise VisualInputError(f"Paired comparison has invalid {column}")

    bounded_columns = {
        "qut_whole_grid_cosine": (-1.0, 1.0),
        "spline_whole_grid_cosine": (-1.0, 1.0),
        "qut_moderately_penalized_whole_grid_cosine": (0.0, 1.0),
        "spline_moderately_penalized_whole_grid_cosine": (0.0, 1.0),
        "qut_support_cosine": (-1.0, 1.0),
        "spline_support_cosine": (-1.0, 1.0),
        "qut_outside_energy_fraction": (0.0, 1.0),
        "spline_outside_energy_fraction": (0.0, 1.0),
    }
    for column, (lower, upper) in bounded_columns.items():
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        if (
            not np.isfinite(values).all()
            or np.any(values < lower - 1e-10)
            or np.any(values > upper + 1e-10)
        ):
            raise VisualInputError(f"Paired comparison has invalid values in {column}")

    for method_prefix in ("qut", "spline"):
        whole = frame[f"{method_prefix}_whole_grid_cosine"].to_numpy(float)
        leakage = frame[f"{method_prefix}_outside_energy_fraction"].to_numpy(float)
        recorded = frame[
            f"{method_prefix}_moderately_penalized_whole_grid_cosine"
        ].to_numpy(float)
        expected = np.maximum(0.0, whole) * (1.0 - leakage)
        if not np.allclose(recorded, expected, rtol=1e-10, atol=1e-10):
            raise VisualInputError(
                f"Paired comparison has algebraically inconsistent {method_prefix} "
                "moderate scores"
            )

    tolerance = pd.to_numeric(frame["winner_tolerance"], errors="raise").to_numpy(float)
    delta = frame["spline_moderately_penalized_whole_grid_cosine"].to_numpy(
        float
    ) - frame["qut_moderately_penalized_whole_grid_cosine"].to_numpy(float)
    expected_winner = np.where(
        delta > tolerance, "spline", np.where(delta < -tolerance, "qut", "tie")
    )
    if not np.array_equal(
        frame["moderate_score_winner"].astype(str).str.lower().to_numpy(),
        expected_winner,
    ):
        raise VisualInputError("Paired comparison winner labels are inconsistent")
    return frame, simulation_tag


def _load_planted_profiles(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not path.is_file():
        raise VisualInputError(f"Planted-profile CSV does not exist: {path}")
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        (
            "profile",
            "profile_label",
            "x_bin",
            "y_bin",
            "flat_index",
            "movement_effect",
            "end_effect",
        ),
        "planted-profile CSV",
    )
    if len(frame) != len(PROFILES) * N_CELLS:
        raise VisualInputError("Planted-profile CSV must contain 800 cell rows")

    maps: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for profile in PROFILES:
        rows = frame.loc[frame["profile"].eq(profile)].copy()
        if len(rows) != N_CELLS:
            raise VisualInputError(f"Expected {N_CELLS} planted cells for {profile}")
        for column in ("x_bin", "y_bin", "flat_index", "movement_effect", "end_effect"):
            rows[column] = pd.to_numeric(rows[column], errors="raise")
        rows = rows.sort_values("flat_index", kind="stable")
        expected_index = np.arange(N_CELLS)
        if not np.array_equal(rows["flat_index"].to_numpy(int), expected_index):
            raise VisualInputError(
                f"Invalid flat-index order for planted profile {profile}"
            )
        if not np.array_equal(rows["x_bin"].to_numpy(int), expected_index // NY):
            raise VisualInputError(f"Invalid x-bin order for planted profile {profile}")
        if not np.array_equal(rows["y_bin"].to_numpy(int), expected_index % NY):
            raise VisualInputError(f"Invalid y-bin order for planted profile {profile}")
        movement = rows["movement_effect"].to_numpy(float)
        endings = rows["end_effect"].to_numpy(float)
        if not np.isfinite(movement).all() or not np.isfinite(endings).all():
            raise VisualInputError(f"Non-finite planted values for {profile}")
        if not np.allclose(endings, 0.0, rtol=0.0, atol=1e-15):
            raise VisualInputError(f"Unexpected planted end effects for {profile}")
        maps[profile] = movement.reshape(NX, NY)
        unique_labels = rows["profile_label"].astype(str).unique().tolist()
        if len(unique_labels) != 1:
            raise VisualInputError(f"Inconsistent profile labels for {profile}")
        labels[profile] = unique_labels[0]
        if labels[profile] != PROFILE_LABELS[profile]:
            raise VisualInputError(f"Unexpected profile label for {profile}")
    return maps, labels


def _metric_values(planted: np.ndarray, recovered: np.ndarray) -> MetricValues:
    planted = np.asarray(planted, dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    if planted.shape != (NX, NY) or recovered.shape != (NX, NY):
        raise VisualInputError(
            f"Map shape mismatch: planted={planted.shape}, recovered={recovered.shape}"
        )
    if not np.isfinite(planted).all() or not np.isfinite(recovered).all():
        raise VisualInputError("Profile maps contain non-finite values")
    support = np.abs(planted) > SUPPORT_TOLERANCE
    if not support.any():
        raise VisualInputError("A planted profile has no supported cells")
    outside = ~support
    background = float(np.median(recovered[outside])) if outside.any() else 0.0
    adjusted = recovered - background
    support_energy = float(np.sum(adjusted[support] ** 2))
    outside_energy = float(np.sum(adjusted[outside] ** 2))
    total_energy = support_energy + outside_energy
    planted_norm = float(np.linalg.norm(planted))
    support_planted_norm = float(np.linalg.norm(planted[support]))
    whole = (
        float(np.sum(planted * adjusted)) / (planted_norm * np.sqrt(total_energy))
        if planted_norm > 0 and total_energy > 0
        else 0.0
    )
    support_cosine = (
        float(np.sum(planted[support] * adjusted[support]))
        / (support_planted_norm * np.sqrt(support_energy))
        if support_planted_norm > 0 and support_energy > 0
        else 0.0
    )
    leakage = outside_energy / total_energy if total_energy > 0 else 0.0
    whole = float(np.clip(whole, -1.0, 1.0))
    support_cosine = float(np.clip(support_cosine, -1.0, 1.0))
    leakage = float(np.clip(leakage, 0.0, 1.0))
    retained = 1.0 - leakage
    nonnegative_whole = max(0.0, whole)
    return MetricValues(
        whole_grid_cosine=whole,
        support_cosine=support_cosine,
        outside_energy_fraction=leakage,
        light_score=float(np.clip(nonnegative_whole * np.sqrt(retained), 0.0, 1.0)),
        moderate_score=float(np.clip(nonnegative_whole * retained, 0.0, 1.0)),
        background=background,
        adjusted=adjusted,
    )


def _assert_metric_match(
    observed: MetricValues,
    row: pd.Series,
    prefix: str,
    run_id: str,
) -> None:
    comparisons = {
        "whole_grid_cosine": observed.whole_grid_cosine,
        "support_cosine": observed.support_cosine,
        "outside_energy_fraction": observed.outside_energy_fraction,
        "moderately_penalized_whole_grid_cosine": observed.moderate_score,
    }
    for suffix, recomputed in comparisons.items():
        recorded = float(row[f"{prefix}_{suffix}"])
        if not np.isclose(recomputed, recorded, rtol=0.0, atol=METRIC_ATOL):
            raise VisualInputError(
                f"{prefix.upper()} map/metric mismatch for {run_id}/{suffix}: "
                f"recomputed={recomputed:.12g}, recorded={recorded:.12g}"
            )


def _npz_scalar(archive: np.lib.npyio.NpzFile, name: str):
    if name not in archive.files:
        raise VisualInputError(f"QUT stream is missing {name!r}")
    value = archive[name]
    if value.shape != ():
        raise VisualInputError(f"QUT stream field {name!r} is not scalar")
    return value.item()


def _load_qut_map(
    stream_root: Path,
    simulation_tag: str,
    profile: str,
    scope: str,
    config: str,
) -> np.ndarray:
    bootstraps, possessions, samples = FROZEN_CONFIGS[config]
    group = f"p{possessions:05d}_s{samples:03d}"
    path = stream_root / profile / f"{profile}__{scope}__{group}.npz"
    if not path.is_file():
        raise VisualInputError(f"QUT nested stream does not exist: {path}")
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise VisualInputError(f"Could not open QUT stream {path}: {exc}") from exc
    with archive_context as archive:
        expected_scalars = {
            "simulation_tag": simulation_tag,
            "profile": profile,
            "scope": scope,
            "bootstrap_group_key": group,
            "possessions_per_bootstrap": possessions,
            "samples_per_possession": samples,
            "map_estimator": "nested_prefix_median_v1",
        }
        for field, expected in expected_scalars.items():
            observed = _npz_scalar(archive, field)
            if str(observed) != str(expected):
                raise VisualInputError(
                    f"QUT stream metadata mismatch in {path}: "
                    f"{field}={observed!r}, expected {expected!r}"
                )
        max_bootstraps = int(_npz_scalar(archive, "max_bootstraps"))
        if max_bootstraps < bootstraps:
            raise VisualInputError(
                f"QUT stream does not contain B={bootstraps}: {path}"
            )
        if "beta_maps" not in archive.files or "map_done" not in archive.files:
            raise VisualInputError(f"QUT stream lacks beta_maps/map_done: {path}")
        beta_maps = np.asarray(archive["beta_maps"][:bootstraps], dtype=float)
        done = np.asarray(archive["map_done"][:bootstraps], dtype=bool)
        if beta_maps.shape != (bootstraps, NX, NY):
            raise VisualInputError(
                f"Unexpected QUT beta-map shape in {path}: {beta_maps.shape}"
            )
        if done.shape != (bootstraps,) or not done.all():
            raise VisualInputError(f"QUT prefix B={bootstraps} is incomplete: {path}")
        if not np.isfinite(beta_maps).all():
            raise VisualInputError(f"QUT beta maps contain non-finite values: {path}")
        return np.median(beta_maps, axis=0)


def _resolve_qut_stream_root(project_root: Path, simulation_tag: str) -> Path:
    """Locate the unique nested-stream directory for a simulation study.

    QUT cache directories append a run-plan fingerprint to ``simulation_tag``;
    the NPZ files themselves retain the unextended scientific study tag.
    """

    parent = project_root / "profile_sim_results" / "nested_streams"
    direct = parent / simulation_tag
    if direct.is_dir():
        return direct
    candidates = sorted(
        path for path in parent.glob(f"{simulation_tag}_plan*") if path.is_dir()
    )
    if len(candidates) != 1:
        raise VisualInputError(
            "Could not uniquely resolve the QUT nested-stream cache for "
            f"{simulation_tag!r}; candidates={[str(path) for path in candidates]}"
        )
    return candidates[0]


R_EXPORT_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Expected RDS path, CSV path, and simulation tag.")
bundle <- readRDS(args[[1]])
if (!is.list(bundle) || is.null(bundle$metadata) || is.null(bundle$maps)) {
  stop("Malformed representative_maps.rds bundle.")
}
if (!identical(as.character(bundle$metadata$simulation_tag), args[[3]])) {
  stop("Spline RDS simulation_tag mismatch.")
}
if (!identical(as.integer(bundle$metadata$shape), c(20L, 10L))) {
  stop("Spline RDS grid shape mismatch.")
}
if (!identical(as.character(bundle$metadata$order), "x-major/y-fast")) {
  stop("Spline RDS cell ordering mismatch.")
}
if (!identical(as.character(bundle$metadata$gam_method), "GCV.Cp")) {
  stop("Spline RDS GAM method mismatch.")
}
if (!identical(as.integer(bundle$metadata$basis_k), 20L)) {
  stop("Spline RDS basis dimension mismatch.")
}
maps <- bundle$maps
if (length(maps) != 100L || is.null(names(maps)) || anyDuplicated(names(maps))) {
  stop("Spline RDS must contain 100 uniquely named maps.")
}
connection <- file(args[[2]], open = "wt", encoding = "UTF-8")
on.exit(close(connection), add = TRUE)
writeLines("run_id,x_bin,y_bin,value", connection)
for (run_id in names(maps)) {
  map <- maps[[run_id]]
  if (!identical(dim(map), c(20L, 10L)) || any(!is.finite(map))) {
    stop(sprintf("Invalid spline map: %s", run_id))
  }
  rows <- data.frame(
    run_id = rep(run_id, 200L),
    x_bin = rep(0:19, each = 10L),
    y_bin = rep(0:9, times = 20L),
    value = as.vector(t(map)),
    stringsAsFactors = FALSE
  )
  write.table(
    rows, connection, sep = ",", row.names = FALSE, col.names = FALSE,
    quote = TRUE, append = TRUE
  )
}
"""


def _r_version_key(path: Path) -> tuple[int, ...]:
    for parent in path.parents:
        if parent.name.startswith("R-"):
            pieces = parent.name[2:].split(".")
            try:
                return tuple(int(piece) for piece in pieces)
            except ValueError:
                break
    return ()


def _find_rscript(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise VisualInputError(f"Rscript executable does not exist: {candidate}")
        return candidate
    discovered = shutil.which("Rscript")
    if discovered:
        return Path(discovered).resolve()
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = sorted(
        program_files.glob("R/R-*/bin/Rscript.exe"), key=_r_version_key, reverse=True
    )
    if not candidates:
        raise VisualInputError(
            "Could not locate Rscript; pass --rscript with the executable path"
        )
    return candidates[0].resolve()


def _load_spline_maps(
    rds_path: Path,
    rscript_path: Path,
    simulation_tag: str,
    expected_run_ids: set[str],
) -> dict[str, np.ndarray]:
    if not rds_path.is_file():
        raise VisualInputError(
            f"Spline representative-map RDS does not exist: {rds_path}"
        )
    with tempfile.TemporaryDirectory(prefix="qut-spline-map-export-") as temporary:
        temporary_path = Path(temporary)
        helper_path = temporary_path / "export_maps.R"
        csv_path = temporary_path / "spline_maps.csv"
        helper_path.write_text(R_EXPORT_SCRIPT, encoding="utf-8", newline="\n")
        process = subprocess.run(
            [
                str(rscript_path),
                "--vanilla",
                str(helper_path),
                str(rds_path.resolve()),
                str(csv_path),
                simulation_tag,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            details = "\n".join(
                item.strip()
                for item in (process.stdout, process.stderr)
                if item.strip()
            )
            raise VisualInputError(f"R could not export spline maps:\n{details}")
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError) as exc:
            raise VisualInputError(
                f"Could not read temporary spline map CSV: {exc}"
            ) from exc

    _require_columns(frame, ("run_id", "x_bin", "y_bin", "value"), "spline map export")
    if len(frame) != len(expected_run_ids) * N_CELLS:
        raise VisualInputError(
            f"Spline map export must have {len(expected_run_ids) * N_CELLS} rows; "
            f"found {len(frame)}"
        )
    if set(frame["run_id"].astype(str)) != expected_run_ids:
        raise VisualInputError("Spline RDS run IDs do not match the paired comparison")
    maps: dict[str, np.ndarray] = {}
    for run_id, rows in frame.groupby("run_id", sort=False):
        rows = rows.copy()
        for column in ("x_bin", "y_bin", "value"):
            rows[column] = pd.to_numeric(rows[column], errors="raise")
        rows = rows.sort_values(["x_bin", "y_bin"], kind="stable")
        expected_index = np.arange(N_CELLS)
        if len(rows) != N_CELLS:
            raise VisualInputError(f"Spline map {run_id} does not have 200 cells")
        if not np.array_equal(rows["x_bin"].to_numpy(int), expected_index // NY):
            raise VisualInputError(f"Spline map {run_id} has invalid x-bin ordering")
        if not np.array_equal(rows["y_bin"].to_numpy(int), expected_index % NY):
            raise VisualInputError(f"Spline map {run_id} has invalid y-bin ordering")
        values = rows["value"].to_numpy(float)
        if not np.isfinite(values).all():
            raise VisualInputError(f"Spline map {run_id} contains non-finite values")
        maps[str(run_id)] = values.reshape(NX, NY)
    return maps


def _unit_l2(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0:
        raise VisualInputError("Cannot unit-normalize a zero or non-finite map")
    return np.asarray(values, dtype=float) / norm


def _draw_support_outline(
    axis: plt.Axes,
    support: np.ndarray,
    *,
    color: str = "#111111",
    linewidth: float = 1.45,
) -> None:
    support = np.asarray(support, dtype=bool)
    for x_bin, y_bin in np.argwhere(support):
        neighbors = (
            (x_bin - 1, y_bin, ((x_bin, y_bin), (x_bin, y_bin + 1))),
            (x_bin + 1, y_bin, ((x_bin + 1, y_bin), (x_bin + 1, y_bin + 1))),
            (x_bin, y_bin - 1, ((x_bin, y_bin), (x_bin + 1, y_bin))),
            (x_bin, y_bin + 1, ((x_bin, y_bin + 1), (x_bin + 1, y_bin + 1))),
        )
        for neighbor_x, neighbor_y, segment in neighbors:
            outside_grid = not (0 <= neighbor_x < NX and 0 <= neighbor_y < NY)
            if outside_grid or not support[neighbor_x, neighbor_y]:
                (x0, y0), (x1, y1) = segment
                axis.plot(
                    [x0, x1],
                    [y0, y1],
                    color=color,
                    linewidth=linewidth,
                    solid_capstyle="butt",
                    zorder=4,
                )


def _style_court_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, NX)
    axis.set_ylim(0, NY)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks((0, 5, 10, 15, 20))
    axis.set_yticks((0, 5, 10))
    axis.tick_params(labelsize=8, length=2.5)
    for spine in axis.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.7)


def _save_figure(fig: plt.Figure, destination: Path, *, dpi: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}"
    )
    try:
        save_kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
            "format": suffix.lstrip("."),
        }
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
        fig.savefig(temporary, **save_kwargs)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _make_spatial_figure(
    pairs: pd.DataFrame,
    planted_maps: Mapping[str, np.ndarray],
    qut_maps: Mapping[str, np.ndarray],
    spline_maps: Mapping[str, np.ndarray],
    profile_labels: Mapping[str, str],
    court_image: np.ndarray,
    config: str,
    scope: str,
) -> plt.Figure:
    display_maps: dict[tuple[str, str], np.ndarray] = {}
    annotations: dict[tuple[str, str], str] = {}
    support_maps: dict[str, np.ndarray] = {}

    for profile in PROFILES:
        run_id = f"{profile}__{scope}__{config}"
        rows = pairs.loc[pairs["run_id"].eq(run_id)]
        if len(rows) != 1:
            raise VisualInputError(f"Expected one paired metric row for {run_id}")
        row = rows.iloc[0]
        planted = planted_maps[profile]
        support = np.abs(planted) > SUPPORT_TOLERANCE
        support_maps[profile] = support

        qut_metrics = _metric_values(planted, qut_maps[run_id])
        spline_metrics = _metric_values(planted, spline_maps[run_id])
        _assert_metric_match(qut_metrics, row, "qut", run_id)
        _assert_metric_match(spline_metrics, row, "spline", run_id)

        display_maps[(profile, "planted")] = _unit_l2(planted)
        display_maps[(profile, "qut")] = _unit_l2(qut_metrics.adjusted)
        display_maps[(profile, "spline")] = _unit_l2(spline_metrics.adjusted)
        annotations[(profile, "planted")] = f"Supported sectors: {int(support.sum())}"
        annotations[(profile, "qut")] = (
            f"Moderate {qut_metrics.moderate_score:.3f}  |  "
            f"Leakage {qut_metrics.outside_energy_fraction:.1%}\n"
            f"Support cosine {qut_metrics.support_cosine:.3f}"
        )
        annotations[(profile, "spline")] = (
            f"Moderate {spline_metrics.moderate_score:.3f}  |  "
            f"Leakage {spline_metrics.outside_energy_fraction:.1%}\n"
            f"Support cosine {spline_metrics.support_cosine:.3f}"
        )

    limit = max(float(np.max(np.abs(values))) for values in display_maps.values())
    if not np.isfinite(limit) or limit <= 0:
        raise VisualInputError("Spatial display maps have no finite dynamic range")
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(
        len(PROFILES),
        3,
        figsize=(17.2, 11.4),
        sharex=True,
        sharey=True,
    )
    methods = ("planted", "qut", "spline")
    method_titles = ("Planted profile", "QUT", "Thin-plate spline")
    image = None
    for row_index, profile in enumerate(PROFILES):
        for column_index, (method, method_title) in enumerate(
            zip(methods, method_titles)
        ):
            axis = axes[row_index, column_index]
            axis.imshow(
                court_image,
                extent=[0, NX, 0, NY],
                origin="upper",
                aspect="equal",
                alpha=0.50,
                zorder=0,
            )
            image = axis.imshow(
                display_maps[(profile, method)].T,
                origin="lower",
                extent=[0, NX, 0, NY],
                interpolation="nearest",
                cmap="RdBu_r",
                norm=normalization,
                alpha=0.78,
                zorder=2,
            )
            _draw_support_outline(axis, support_maps[profile])
            _style_court_axis(axis)
            if row_index == 0:
                axis.set_title(method_title, fontsize=13, fontweight="semibold", pad=8)
            if column_index == 0:
                axis.set_ylabel(
                    f"{profile_labels[profile]}\ny bin",
                    fontsize=10,
                    fontweight="semibold",
                )
            if row_index == len(PROFILES) - 1:
                axis.set_xlabel("x bin", fontsize=9)
            axis.text(
                0.018,
                0.025,
                annotations[(profile, method)],
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.2,
                color="#111111",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "white",
                    "edgecolor": "#555555",
                    "alpha": 0.90,
                    "linewidth": 0.6,
                },
                zorder=5,
            )

    bootstraps, possessions, samples = FROZEN_CONFIGS[config]
    fig.suptitle(
        "Spatial profile recovery: planted signal, QUT, and spline\n"
        f"{SCOPE_LABELS[scope]} | {config} | B={bootstraps}, "
        f"P={possessions:,}, S={samples}",
        fontsize=15,
        fontweight="semibold",
        y=0.991,
        linespacing=1.25,
    )
    fig.text(
        0.5,
        0.914,
        "Recovered maps subtract the off-support median; all panels are unit-L2 "
        "normalized for shape comparison. Black line = planted support.",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.09, right=0.92, top=0.855, bottom=0.06, hspace=0.34, wspace=0.12
    )
    if image is None:
        raise AssertionError("Spatial figure did not create a heatmap")
    colorbar_axis = fig.add_axes([0.935, 0.15, 0.015, 0.66])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Unit-L2 map value (shape only)", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
    fig.text(
        0.5,
        0.012,
        "Display normalization does not affect the annotated v2 similarity or leakage metrics.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    return fig


def _mean_box_text(frame: pd.DataFrame, metric: str, *, percent: bool) -> str:
    qut = float(frame[f"qut_{metric}"].mean())
    spline = float(frame[f"spline_{metric}"].mean())
    if percent:
        return f"Mean QUT {qut:.1%}\nMean spline {spline:.1%}"
    return f"Mean QUT {qut:.3f}\nMean spline {spline:.3f}"


def _make_paired_figure(pairs: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.2))
    panels = (
        (
            "moderately_penalized_whole_grid_cosine",
            "Moderately penalized whole-grid cosine",
            "Higher is better",
            False,
            "QUT moderate score",
            "Spline moderate score",
        ),
        (
            "outside_energy_fraction",
            "Outside-support energy",
            "Lower is better",
            True,
            "QUT outside-support energy",
            "Spline outside-support energy",
        ),
    )

    for axis, (metric, title, direction, percent, x_label, y_label) in zip(
        axes, panels
    ):
        for profile in PROFILES:
            for scope in SCOPES:
                rows = pairs.loc[
                    pairs["profile"].eq(profile) & pairs["scope"].eq(scope)
                ]
                axis.scatter(
                    rows[f"qut_{metric}"].to_numpy(float),
                    rows[f"spline_{metric}"].to_numpy(float),
                    s=51,
                    marker=SCOPE_MARKERS[scope],
                    color=PROFILE_COLORS[profile],
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.82,
                    zorder=3,
                )
        axis.plot(
            [0, 1],
            [0, 1],
            linestyle=(0, (5, 4)),
            color="#555555",
            linewidth=1.2,
            zorder=1,
        )
        axis.text(
            0.51,
            0.49,
            "equal performance",
            transform=axis.transAxes,
            rotation=45,
            ha="center",
            va="center",
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            zorder=2,
        )
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(x_label, fontsize=10)
        axis.set_ylabel(y_label, fontsize=10)
        axis.set_title(f"{title}\n{direction}", fontsize=13, fontweight="semibold")
        axis.grid(True, color="#d7d7d7", linewidth=0.65, alpha=0.75, zorder=0)
        axis.tick_params(labelsize=9)
        mean_box_x = 0.965 if percent else 0.035
        axis.text(
            mean_box_x,
            0.965,
            _mean_box_text(pairs, metric, percent=percent),
            transform=axis.transAxes,
            ha="right" if percent else "left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": "#777777",
                "alpha": 0.93,
                "linewidth": 0.65,
            },
            zorder=5,
        )

    score_delta = pairs["spline_moderately_penalized_whole_grid_cosine"].to_numpy(
        float
    ) - pairs["qut_moderately_penalized_whole_grid_cosine"].to_numpy(float)
    leakage_delta = pairs["spline_outside_energy_fraction"].to_numpy(float) - pairs[
        "qut_outside_energy_fraction"
    ].to_numpy(float)
    score_qut_wins = int(np.sum(score_delta < -1e-9))
    leakage_qut_wins = int(np.sum(leakage_delta > 1e-9))
    axes[0].text(
        0.965,
        0.965,
        f"QUT higher: {score_qut_wins}/{len(pairs)}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        fontweight="semibold",
        color="#222222",
    )
    axes[1].text(
        0.965,
        0.035,
        f"QUT lower: {leakage_qut_wins}/{len(pairs)}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        fontweight="semibold",
        color="#222222",
    )

    profile_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PROFILE_COLORS[profile],
            markeredgecolor="white",
            markersize=8,
            label=PROFILE_LABELS[profile],
        )
        for profile in PROFILES
    ]
    scope_handles = [
        Line2D(
            [0],
            [0],
            marker=SCOPE_MARKERS[scope],
            linestyle="none",
            markerfacecolor="#707070",
            markeredgecolor="white",
            markersize=8,
            label=SCOPE_LABELS[scope],
        )
        for scope in SCOPES
    ]
    profile_legend = fig.legend(
        handles=profile_handles,
        title="Planted profile (color)",
        loc="lower center",
        bbox_to_anchor=(0.29, 0.015),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
    )
    fig.add_artist(profile_legend)
    fig.legend(
        handles=scope_handles,
        title="Evaluation scope (marker)",
        loc="lower center",
        bbox_to_anchor=(0.75, 0.015),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
    )
    fig.suptitle(
        "QUT versus thin-plate splines across 100 matched profile-recovery cases\n"
        "Five frozen settings × four planted profiles × five evaluation scopes",
        fontsize=15,
        fontweight="semibold",
        y=0.985,
        linespacing=1.25,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.20, wspace=0.20)
    return fig


def _write_manifest(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    destination = output_dir / "visual_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fieldnames = (
        "figure",
        "format",
        "relative_path",
        "config_key",
        "scope",
        "n_paired_cases",
        "normalization",
        "similarity_method",
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def generate_visuals(
    *,
    project_root: Path,
    pairs_path: Path,
    planted_path: Path,
    spline_rds_path: Path,
    qut_stream_root: Path | None,
    court_path: Path,
    output_dir: Path,
    config: str,
    scope: str,
    rscript: Path | None,
    dpi: int,
) -> tuple[Path, ...]:
    if config not in FROZEN_CONFIGS:
        raise VisualInputError(f"Unknown frozen configuration: {config}")
    if scope not in SCOPES:
        raise VisualInputError(f"Unknown evaluation scope: {scope}")
    if dpi < 100 or dpi > 600:
        raise VisualInputError("DPI must be between 100 and 600")

    pairs, simulation_tag = _validate_pairs(pairs_path)
    planted_maps, profile_labels = _load_planted_profiles(planted_path)
    expected_run_ids = set(pairs["run_id"].astype(str))
    rscript_path = _find_rscript(rscript)
    spline_maps = _load_spline_maps(
        spline_rds_path, rscript_path, simulation_tag, expected_run_ids
    )

    if qut_stream_root is None:
        qut_stream_root = _resolve_qut_stream_root(project_root, simulation_tag)
    if not qut_stream_root.is_dir():
        raise VisualInputError(
            f"QUT nested-stream root does not exist: {qut_stream_root}"
        )
    if not court_path.is_file():
        raise VisualInputError(f"Court image does not exist: {court_path}")
    court_image = mpimg.imread(court_path)

    qut_maps: dict[str, np.ndarray] = {}
    for profile in PROFILES:
        run_id = f"{profile}__{scope}__{config}"
        qut_maps[run_id] = _load_qut_map(
            qut_stream_root, simulation_tag, profile, scope, config
        )

    spatial = _make_spatial_figure(
        pairs,
        planted_maps,
        qut_maps,
        spline_maps,
        profile_labels,
        court_image,
        config,
        scope,
    )
    aggregate = _make_paired_figure(pairs)

    spatial_stem = f"qut_vs_spline_spatial__{config}__{scope}"
    aggregate_stem = "qut_vs_spline_paired_metrics__all100"
    outputs: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    try:
        for figure_name, figure, stem in (
            ("spatial_profile_plate", spatial, spatial_stem),
            ("paired_metric_scatter", aggregate, aggregate_stem),
        ):
            for extension in (".png", ".pdf"):
                destination = output_dir / f"{stem}{extension}"
                _save_figure(figure, destination, dpi=dpi)
                outputs.append(destination)
                manifest_rows.append(
                    {
                        "figure": figure_name,
                        "format": extension.lstrip("."),
                        "relative_path": destination.name,
                        "config_key": (
                            config
                            if figure_name == "spatial_profile_plate"
                            else "all_frozen"
                        ),
                        "scope": (
                            scope
                            if figure_name == "spatial_profile_plate"
                            else "all_scopes"
                        ),
                        "n_paired_cases": (
                            len(PROFILES)
                            if figure_name == "spatial_profile_plate"
                            else len(pairs)
                        ),
                        "normalization": (
                            "off-support-median-adjusted_then_unit-L2_for_display"
                            if figure_name == "spatial_profile_plate"
                            else "none"
                        ),
                        "similarity_method": SIMILARITY_METHOD,
                    }
                )
    finally:
        plt.close(spatial)
        plt.close(aggregate)

    manifest = _write_manifest(output_dir, manifest_rows)
    outputs.append(manifest)
    return tuple(outputs)


def _parser() -> argparse.ArgumentParser:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Render validated QUT-versus-spline profile-recovery figures."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--planted-profiles", type=Path, default=None)
    parser.add_argument("--spline-rds", type=Path, default=None)
    parser.add_argument(
        "--qut-stream-root",
        type=Path,
        default=None,
        help="Override the nested QUT stream root; normally derived from simulation_tag.",
    )
    parser.add_argument("--court-image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--config-key", choices=tuple(FROZEN_CONFIGS), default=DEFAULT_CONFIG
    )
    parser.add_argument("--scope", choices=SCOPES, default=DEFAULT_SCOPE)
    parser.add_argument("--rscript", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    study_root = (
        project_root
        / "profile_sim_results"
        / "spline_comparison"
        / "games10_top5_tp_k20"
    )
    pairs_path = args.pairs or (
        study_root / "comparison" / "qut_vs_spline_profile_scope.csv"
    )
    planted_path = (
        args.planted_profiles or study_root / "inputs" / "planted_profiles.csv"
    )
    spline_rds_path = (
        args.spline_rds or study_root / "spline_gcv" / "representative_maps.rds"
    )
    court_path = args.court_image or project_root / "court.jpg"
    output_dir = (
        args.output_dir or project_root / "results" / "qut-vs-spline-profile-recovery"
    )
    try:
        outputs = generate_visuals(
            project_root=project_root,
            pairs_path=pairs_path.resolve(),
            planted_path=planted_path.resolve(),
            spline_rds_path=spline_rds_path.resolve(),
            qut_stream_root=(
                args.qut_stream_root.resolve() if args.qut_stream_root else None
            ),
            court_path=court_path.resolve(),
            output_dir=output_dir.resolve(),
            config=args.config_key,
            scope=args.scope,
            rscript=args.rscript,
            dpi=args.dpi,
        )
    except (VisualInputError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
