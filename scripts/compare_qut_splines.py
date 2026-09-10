"""Create paired descriptive comparisons of QUT and spline profile recovery.

The five configurations are intentionally frozen from the ten-game QUT ranking.
An invocation compares one study at a time (the ten-game study or its optional
eight-game held-out replication) and requires complete, one-to-one coverage of
all four profiles and all five evaluation scopes. Both inputs must identify the
same simulation and the adjusted v2 whole-grid metric definition.

Example
-------
python scripts/compare_qut_splines.py \
    --qut-metrics path/to/qut_metrics.csv \
    --spline-metrics path/to/spline_metrics.csv \
    --output-dir path/to/comparison
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# These settings are frozen from the current ten-game QUT ranking. Their order
# is the QUT rank order and is retained in every published comparison table.
FROZEN_CONFIGS: tuple[tuple[str, int, int, int], ...] = (
    ("b050_p02500_s250", 50, 2500, 250),
    ("b050_p01000_s250", 50, 1000, 250),
    ("b100_p02500_s250", 100, 2500, 250),
    ("b100_p01000_s200", 100, 1000, 200),
    ("b100_p01000_s250", 100, 1000, 250),
)
PROFILES: tuple[str, ...] = (
    "reference",
    "high_y_side",
    "low_y_side",
    "perimeter",
)
SCOPES: tuple[str, ...] = (
    "full_game",
    "quarter_1",
    "quarter_2",
    "quarter_3",
    "quarter_4",
)
SETTING_COLUMNS: tuple[str, ...] = (
    "n_bootstraps",
    "possessions_per_bootstrap",
    "samples_per_possession",
)
PAIR_COLUMNS: tuple[str, ...] = (
    "run_id",
    "profile",
    "scope",
    "config_key",
    *SETTING_COLUMNS,
)
METRIC_COLUMNS: tuple[str, ...] = (
    "whole_grid_cosine",
    "lightly_penalized_whole_grid_cosine",
    "moderately_penalized_whole_grid_cosine",
    "support_cosine",
    "outside_energy_fraction",
)
METRIC_DEFINITION_COLUMNS: tuple[str, ...] = (
    "similarity_method",
    "light_leakage_exponent",
    "moderate_leakage_exponent",
)
SUCCESS_COLUMNS: tuple[str, ...] = ("map_status", "status")
EXPECTED_SIMILARITY_METHOD = "whole_grid_cosine_dual_leakage_penalty_v2"
EXPECTED_LIGHT_LEAKAGE_EXPONENT = 0.5
EXPECTED_MODERATE_LEAKAGE_EXPONENT = 1.0
FORMULA_ABSOLUTE_TOLERANCE = 1e-10
FORMULA_RELATIVE_TOLERANCE = 1e-10
METRIC_BOUNDS: Mapping[str, tuple[float, float]] = {
    "whole_grid_cosine": (-1.0, 1.0),
    "lightly_penalized_whole_grid_cosine": (0.0, 1.0),
    "moderately_penalized_whole_grid_cosine": (0.0, 1.0),
    "support_cosine": (-1.0, 1.0),
    "outside_energy_fraction": (0.0, 1.0),
}
EXPECTED_ROWS = len(FROZEN_CONFIGS) * len(PROFILES) * len(SCOPES)
SELECTION_LABEL = "QUT ten-game moderate ranking"
DELTA_DEFINITION = "spline_minus_qut"
DEPENDENCE_CAVEAT = (
    "Descriptive paired comparison only: repeated profile/scope evaluations "
    "and QUT-selected configurations are not independent inferential replicates."
)
SELECTION_CAVEAT = (
    "Configurations were selected using the ten-game QUT moderate-score "
    "ranking; all-config differences therefore describe these frozen settings "
    "and are not an independent model-selection comparison."
)
OUTPUT_FILENAMES: tuple[str, ...] = (
    "qut_vs_spline_profile_scope.csv",
    "qut_vs_spline_overall.csv",
    "qut_vs_spline_descriptive_summary.csv",
)


class ComparisonInputError(ValueError):
    """Raised when a metrics input cannot support the paired comparison."""


def _duplicate_names(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _read_metrics(path: Path, method: str) -> pd.DataFrame:
    if not path.is_file():
        raise ComparisonInputError(f"{method} metrics file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except StopIteration as exc:
        raise ComparisonInputError(f"{method} metrics file is empty: {path}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ComparisonInputError(
            f"Could not read the {method} metrics header from {path}: {exc}"
        ) from exc

    duplicate_columns = _duplicate_names(header)
    if duplicate_columns:
        raise ComparisonInputError(
            f"{method} metrics has duplicate column names: {duplicate_columns}"
        )

    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ComparisonInputError(
            f"Could not read {method} metrics CSV {path}: {exc}"
        ) from exc

    required = (
        set(PAIR_COLUMNS)
        | set(METRIC_COLUMNS)
        | set(METRIC_DEFINITION_COLUMNS)
        | {"simulation_tag"}
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ComparisonInputError(
            f"{method} metrics is missing required columns: {missing}"
        )
    if not any(column in frame.columns for column in SUCCESS_COLUMNS):
        raise ComparisonInputError(
            f"{method} metrics must contain a success-status column; expected "
            f"one of {list(SUCCESS_COLUMNS)}"
        )
    return frame


def _format_key_examples(keys: Iterable[tuple[object, ...]], limit: int = 5) -> str:
    ordered = sorted((tuple(map(str, key)) for key in keys))
    preview = ", ".join("/".join(key) for key in ordered[:limit])
    if len(ordered) > limit:
        preview += f", ... ({len(ordered)} total)"
    return preview or "none"


def _expected_identity_keys() -> set[tuple[str, str, str, str]]:
    return {
        (f"{profile}__{scope}__{config_key}", profile, scope, config_key)
        for (config_key, _, _, _), profile, scope in product(
            FROZEN_CONFIGS, PROFILES, SCOPES
        )
    }


def _coerce_and_validate_metrics(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = frame.copy()

    if len(frame) != EXPECTED_ROWS:
        raise ComparisonInputError(
            f"{method} metrics must contain exactly {EXPECTED_ROWS} rows "
            f"(5 configs x 4 profiles x 5 scopes); found {len(frame)}"
        )

    string_columns = ("run_id", "profile", "scope", "config_key")
    for column in string_columns:
        if frame[column].isna().any():
            raise ComparisonInputError(
                f"{method} metrics column {column!r} contains missing values"
            )
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ComparisonInputError(
                f"{method} metrics column {column!r} contains empty values"
            )

    frame["simulation_tag"] = frame["simulation_tag"].astype("string").str.strip()
    if frame["simulation_tag"].isna().any() or frame["simulation_tag"].eq("").any():
        raise ComparisonInputError(
            f"{method} metrics simulation_tag contains missing or empty values"
        )
    simulation_tags = frame["simulation_tag"].unique().tolist()
    if len(simulation_tags) != 1:
        raise ComparisonInputError(
            f"{method} metrics must identify exactly one simulation_tag; "
            f"found {simulation_tags}"
        )

    frame["similarity_method"] = (
        frame["similarity_method"].astype("string").str.strip()
    )
    if (
        frame["similarity_method"].isna().any()
        or frame["similarity_method"].eq("").any()
    ):
        raise ComparisonInputError(
            f"{method} metrics similarity_method contains missing or empty values"
        )
    observed_methods = frame["similarity_method"].unique().tolist()
    if observed_methods != [EXPECTED_SIMILARITY_METHOD]:
        raise ComparisonInputError(
            f"{method} metrics must use similarity_method "
            f"{EXPECTED_SIMILARITY_METHOD!r}; found {observed_methods}"
        )

    present_status_columns = [
        column for column in SUCCESS_COLUMNS if column in frame.columns
    ]
    for column in present_status_columns:
        status = frame[column].astype("string").str.strip().str.lower()
        if status.isna().any() or status.eq("").any():
            raise ComparisonInputError(
                f"{method} metrics column {column!r} contains missing or empty values"
            )
        failures = ~status.eq("ok")
        if failures.any():
            examples = sorted(status.loc[failures].unique().tolist())
            raise ComparisonInputError(
                f"{method} metrics contains unsuccessful {column} values: {examples}"
            )
        frame[column] = status

    expected_exponents = {
        "light_leakage_exponent": EXPECTED_LIGHT_LEAKAGE_EXPONENT,
        "moderate_leakage_exponent": EXPECTED_MODERATE_LEAKAGE_EXPONENT,
    }
    for column, expected in expected_exponents.items():
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ComparisonInputError(
                f"{method} metrics column {column!r} must be numeric"
            ) from exc
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ComparisonInputError(
                f"{method} metrics column {column!r} contains non-finite values"
            )
        matches = np.isclose(
            numeric,
            expected,
            rtol=0.0,
            atol=FORMULA_ABSOLUTE_TOLERANCE,
        )
        if not matches.all():
            examples = np.unique(numeric[~matches])[:5].tolist()
            raise ComparisonInputError(
                f"{method} metrics column {column!r} must equal {expected}; "
                f"found {examples}"
            )
        frame[column] = numeric

    for column in SETTING_COLUMNS:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ComparisonInputError(
                f"{method} metrics column {column!r} must be numeric"
            ) from exc
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ComparisonInputError(
                f"{method} metrics column {column!r} must contain finite integers"
            )
        frame[column] = numeric.astype(np.int64)

    expected_settings = {
        config_key: (n_bootstraps, possessions, samples)
        for config_key, n_bootstraps, possessions, samples in FROZEN_CONFIGS
    }
    actual_configs = set(frame["config_key"])
    frozen_configs = set(expected_settings)
    if actual_configs != frozen_configs:
        raise ComparisonInputError(
            f"{method} metrics config_key values do not equal the frozen "
            f"ten-game top five; missing={sorted(frozen_configs - actual_configs)}, "
            f"extra={sorted(actual_configs - frozen_configs)}"
        )

    for config_key, settings in expected_settings.items():
        rows = frame.loc[frame["config_key"].eq(config_key), SETTING_COLUMNS]
        matches = np.column_stack(
            [rows[column].to_numpy() == expected for column, expected in zip(SETTING_COLUMNS, settings)]
        ).all(axis=1)
        if not matches.all():
            bad_settings = rows.loc[~matches].drop_duplicates().to_dict("records")
            raise ComparisonInputError(
                f"{method} metrics has settings inconsistent with {config_key}: "
                f"expected {dict(zip(SETTING_COLUMNS, settings))}, found {bad_settings}"
            )

    duplicate_mask = frame.duplicated(list(PAIR_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicate_keys = {
            tuple(row)
            for row in frame.loc[duplicate_mask, PAIR_COLUMNS].itertuples(
                index=False, name=None
            )
        }
        raise ComparisonInputError(
            f"{method} metrics has duplicate pairing keys: "
            f"{_format_key_examples(duplicate_keys)}"
        )

    actual_identity = {
        tuple(row)
        for row in frame.loc[:, ["run_id", "profile", "scope", "config_key"]].itertuples(
            index=False, name=None
        )
    }
    expected_identity = _expected_identity_keys()
    if actual_identity != expected_identity:
        missing = expected_identity - actual_identity
        extra = actual_identity - expected_identity
        raise ComparisonInputError(
            f"{method} metrics does not contain the exact frozen profile/scope grid; "
            f"missing={_format_key_examples(missing)}, "
            f"extra={_format_key_examples(extra)}"
        )

    bounds_epsilon = 1e-10
    for column in METRIC_COLUMNS:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ComparisonInputError(
                f"{method} metrics column {column!r} must be numeric"
            ) from exc
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ComparisonInputError(
                f"{method} metrics column {column!r} contains missing or non-finite values"
            )
        lower, upper = METRIC_BOUNDS[column]
        outside_bounds = (numeric < lower - bounds_epsilon) | (
            numeric > upper + bounds_epsilon
        )
        if outside_bounds.any():
            examples = numeric[outside_bounds][:5].tolist()
            raise ComparisonInputError(
                f"{method} metrics column {column!r} must be in [{lower}, {upper}]; "
                f"found {examples}"
            )
        frame[column] = numeric

    whole = frame["whole_grid_cosine"].to_numpy(dtype=float)
    leakage = frame["outside_energy_fraction"].to_numpy(dtype=float)
    retained = np.clip(1.0 - leakage, 0.0, 1.0)
    nonnegative_whole = np.maximum(0.0, whole)
    expected_formula_values = {
        "lightly_penalized_whole_grid_cosine": np.clip(
            nonnegative_whole
            * retained**EXPECTED_LIGHT_LEAKAGE_EXPONENT,
            0.0,
            1.0,
        ),
        "moderately_penalized_whole_grid_cosine": np.clip(
            nonnegative_whole
            * retained**EXPECTED_MODERATE_LEAKAGE_EXPONENT,
            0.0,
            1.0,
        ),
    }
    for column, expected in expected_formula_values.items():
        observed = frame[column].to_numpy(dtype=float)
        matches = np.isclose(
            observed,
            expected,
            rtol=FORMULA_RELATIVE_TOLERANCE,
            atol=FORMULA_ABSOLUTE_TOLERANCE,
        )
        if not matches.all():
            bad_indices = np.flatnonzero(~matches)[:5]
            examples = [
                {
                    "run_id": str(frame.iloc[index]["run_id"]),
                    "observed": float(observed[index]),
                    "expected": float(expected[index]),
                }
                for index in bad_indices
            ]
            raise ComparisonInputError(
                f"{method} metrics column {column!r} is inconsistent with "
                "whole_grid_cosine and outside_energy_fraction under the "
                f"v2 definition; examples={examples}"
            )

    return frame


def _single_simulation_tag(frame: pd.DataFrame, method: str) -> str:
    """Return the exact study identity after frame-level validation."""

    tags = frame["simulation_tag"].unique().tolist()
    if len(tags) != 1:
        raise ComparisonInputError(
            f"{method} metrics must identify exactly one simulation_tag; found {tags}"
        )
    return str(tags[0])


def _single_study_size(frame: pd.DataFrame, method: str) -> int | None:
    """Infer an optional study size while allowing minimal metrics-only CSVs."""

    candidates: list[int] = []
    if "simulated_games" in frame.columns:
        nonmissing = frame["simulated_games"].dropna()
        if len(nonmissing) != len(frame):
            raise ComparisonInputError(
                f"{method} metrics simulated_games is present but has missing values"
            )
        try:
            numeric = pd.to_numeric(nonmissing, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ComparisonInputError(
                f"{method} metrics simulated_games must be 8 or 10"
            ) from exc
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ComparisonInputError(
                f"{method} metrics simulated_games must contain finite integers"
            )
        candidates.extend(np.unique(numeric.astype(int)).tolist())

    if "simulation_tag" in frame.columns:
        tags = frame["simulation_tag"].dropna().astype(str)
        extracted = tags.str.extract(r"(?:^|_)games(8|10)(?:_|$)", expand=False).dropna()
        candidates.extend(extracted.astype(int).unique().tolist())

    unique = sorted(set(candidates))
    if not unique:
        return None
    if len(unique) != 1 or unique[0] not in (8, 10):
        raise ComparisonInputError(
            f"{method} metrics mixes or identifies unsupported simulated-game "
            f"counts: {unique}; expected one of [8, 10]"
        )
    return unique[0]


def _winner(delta: np.ndarray | pd.Series | float, tolerance: float):
    values = np.asarray(delta, dtype=float)
    winners = np.where(
        values > tolerance,
        "spline",
        np.where(values < -tolerance, "qut", "tie"),
    )
    if winners.ndim == 0:
        return str(winners.item())
    return winners


def _build_profile_scope_table(
    qut: pd.DataFrame,
    spline: pd.DataFrame,
    simulation_tag: str,
    simulated_games: int | None,
    tolerance: float,
) -> pd.DataFrame:
    qut_columns = list(PAIR_COLUMNS) + list(METRIC_COLUMNS)
    spline_columns = list(PAIR_COLUMNS) + list(METRIC_COLUMNS)
    paired = qut.loc[:, qut_columns].merge(
        spline.loc[:, spline_columns],
        how="inner",
        on=list(PAIR_COLUMNS),
        suffixes=("_qut", "_spline"),
        validate="one_to_one",
        sort=False,
    )
    if len(paired) != EXPECTED_ROWS:
        raise ComparisonInputError(
            f"The QUT/spline inner merge produced {len(paired)} rows; "
            f"expected {EXPECTED_ROWS} complete pairs"
        )

    output = paired.loc[:, PAIR_COLUMNS].copy()
    output.insert(0, "moderate_leakage_exponent", EXPECTED_MODERATE_LEAKAGE_EXPONENT)
    output.insert(0, "light_leakage_exponent", EXPECTED_LIGHT_LEAKAGE_EXPONENT)
    output.insert(0, "similarity_method", EXPECTED_SIMILARITY_METHOD)
    output.insert(0, "settings_selected_by", SELECTION_LABEL)
    output.insert(0, "simulated_games", simulated_games if simulated_games else "")
    output.insert(0, "simulation_tag", simulation_tag)
    output.insert(
        output.columns.get_loc("config_key") + 1,
        "config_selection_rank",
        output["config_key"].map(
            {config[0]: rank for rank, config in enumerate(FROZEN_CONFIGS, start=1)}
        ),
    )

    for metric in METRIC_COLUMNS:
        qut_column = f"{metric}_qut"
        spline_column = f"{metric}_spline"
        output[f"qut_{metric}"] = paired[qut_column].to_numpy()
        output[f"spline_{metric}"] = paired[spline_column].to_numpy()
        output[f"delta_{metric}"] = (
            paired[spline_column].to_numpy() - paired[qut_column].to_numpy()
        )

    moderate_delta = output["delta_moderately_penalized_whole_grid_cosine"]
    output["moderate_score_winner"] = _winner(moderate_delta, tolerance)
    output["winner_tolerance"] = tolerance
    output["delta_definition"] = DELTA_DEFINITION

    config_order = {config[0]: index for index, config in enumerate(FROZEN_CONFIGS)}
    profile_order = {value: index for index, value in enumerate(PROFILES)}
    scope_order = {value: index for index, value in enumerate(SCOPES)}
    output = (
        output.assign(
            _config_order=output["config_key"].map(config_order),
            _profile_order=output["profile"].map(profile_order),
            _scope_order=output["scope"].map(scope_order),
        )
        .sort_values(
            ["_config_order", "_profile_order", "_scope_order"], kind="stable"
        )
        .drop(columns=["_config_order", "_profile_order", "_scope_order"])
        .reset_index(drop=True)
    )
    return output


def _summary_values(frame: pd.DataFrame, tolerance: float) -> dict[str, object]:
    values: dict[str, object] = {}
    for metric in METRIC_COLUMNS:
        values[f"mean_qut_{metric}"] = frame[f"qut_{metric}"].mean()
        values[f"mean_spline_{metric}"] = frame[f"spline_{metric}"].mean()
        # Taking the mean of paired differences keeps the direction explicit.
        values[f"mean_delta_{metric}"] = frame[f"delta_{metric}"].mean()

    winner_counts = frame["moderate_score_winner"].value_counts()
    values["spline_moderate_wins"] = int(winner_counts.get("spline", 0))
    values["qut_moderate_wins"] = int(winner_counts.get("qut", 0))
    values["moderate_ties"] = int(winner_counts.get("tie", 0))
    values["mean_moderate_score_winner"] = _winner(
        values["mean_delta_moderately_penalized_whole_grid_cosine"], tolerance
    )
    return values


def _build_per_config_table(
    detail: pd.DataFrame,
    simulation_tag: str,
    simulated_games: int | None,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank, (config_key, n_bootstraps, possessions, samples) in enumerate(
        FROZEN_CONFIGS, start=1
    ):
        group = detail.loc[detail["config_key"].eq(config_key)]
        if len(group) != len(PROFILES) * len(SCOPES):
            raise ComparisonInputError(
                f"Internal coverage error for {config_key}: found {len(group)} pairs"
        )
        row: dict[str, object] = {
            "simulation_tag": simulation_tag,
            "simulated_games": simulated_games if simulated_games else "",
            "settings_selected_by": SELECTION_LABEL,
            "similarity_method": EXPECTED_SIMILARITY_METHOD,
            "light_leakage_exponent": EXPECTED_LIGHT_LEAKAGE_EXPONENT,
            "moderate_leakage_exponent": EXPECTED_MODERATE_LEAKAGE_EXPONENT,
            "config_selection_rank": rank,
            "config_key": config_key,
            "n_bootstraps": n_bootstraps,
            "possessions_per_bootstrap": possessions,
            "samples_per_possession": samples,
            "n_profile_scope_pairs": len(group),
            "n_profiles": group["profile"].nunique(),
            "n_scopes": group["scope"].nunique(),
        }
        row.update(_summary_values(group, tolerance))
        row["winner_tolerance"] = tolerance
        row["delta_definition"] = DELTA_DEFINITION
        row["comparison_type"] = "paired_descriptive"
        rows.append(row)
    return pd.DataFrame(rows)


def _build_descriptive_summary(
    detail: pd.DataFrame,
    simulation_tag: str,
    simulated_games: int | None,
    tolerance: float,
) -> pd.DataFrame:
    row: dict[str, object] = {
        "summary_level": "all_frozen_config_profile_scope_pairs",
        "simulation_tag": simulation_tag,
        "simulated_games": simulated_games if simulated_games else "",
        "settings_selected_by": SELECTION_LABEL,
        "similarity_method": EXPECTED_SIMILARITY_METHOD,
        "light_leakage_exponent": EXPECTED_LIGHT_LEAKAGE_EXPONENT,
        "moderate_leakage_exponent": EXPECTED_MODERATE_LEAKAGE_EXPONENT,
        "n_configurations": detail["config_key"].nunique(),
        "n_profiles": detail["profile"].nunique(),
        "n_scopes": detail["scope"].nunique(),
        "n_profile_scope_pairs": len(detail),
    }
    row.update(_summary_values(detail, tolerance))
    row.update(
        {
            "winner_tolerance": tolerance,
            "delta_definition": DELTA_DEFINITION,
            "comparison_type": "paired_descriptive",
            "inferential_test_performed": False,
            "selection_caveat": SELECTION_CAVEAT,
            "dependence_caveat": DEPENDENCE_CAVEAT,
        }
    )
    return pd.DataFrame([row])


def _write_csvs_atomically(
    output_dir: Path, frames: Sequence[tuple[str, pd.DataFrame]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for filename, frame in frames:
            destination = output_dir / filename
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{filename}.",
                suffix=".tmp",
                dir=output_dir,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                staged.append((temporary, destination))
                frame.to_csv(handle, index=False, lineterminator="\n")
                handle.flush()
                os.fsync(handle.fileno())

        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def compare_metrics(
    qut_metrics: Path,
    spline_metrics: Path,
    output_dir: Path,
    winner_tolerance: float = 1e-9,
) -> tuple[Path, Path, Path]:
    """Validate, pair, summarize, and atomically publish one study comparison."""

    if not np.isfinite(winner_tolerance) or winner_tolerance < 0:
        raise ComparisonInputError("winner tolerance must be finite and non-negative")

    qut = _coerce_and_validate_metrics(_read_metrics(qut_metrics, "QUT"), "QUT")
    spline = _coerce_and_validate_metrics(
        _read_metrics(spline_metrics, "spline"), "spline"
    )

    qut_tag = _single_simulation_tag(qut, "QUT")
    spline_tag = _single_simulation_tag(spline, "spline")
    if qut_tag != spline_tag:
        raise ComparisonInputError(
            "QUT and spline metrics have different simulation_tag values: "
            f"{qut_tag!r} versus {spline_tag!r}"
        )

    qut_games = _single_study_size(qut, "QUT")
    spline_games = _single_study_size(spline, "spline")
    if qut_games is not None and spline_games is not None and qut_games != spline_games:
        raise ComparisonInputError(
            "QUT and spline metrics are from different studies: "
            f"{qut_games} versus {spline_games} simulated games"
        )
    simulated_games = qut_games if qut_games is not None else spline_games

    detail = _build_profile_scope_table(
        qut, spline, qut_tag, simulated_games, winner_tolerance
    )
    per_config = _build_per_config_table(
        detail, qut_tag, simulated_games, winner_tolerance
    )
    descriptive = _build_descriptive_summary(
        detail, qut_tag, simulated_games, winner_tolerance
    )

    frames = tuple(zip(OUTPUT_FILENAMES, (detail, per_config, descriptive)))
    try:
        _write_csvs_atomically(output_dir, frames)
    except OSError as exc:
        raise ComparisonInputError(
            f"Could not publish comparison CSVs under {output_dir}: {exc}"
        ) from exc
    outputs = tuple(output_dir / filename for filename in OUTPUT_FILENAMES)
    return outputs[0], outputs[1], outputs[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare complete QUT and spline metrics for the frozen five "
            "profile-recovery configurations."
        )
    )
    parser.add_argument("--qut-metrics", required=True, type=Path)
    parser.add_argument("--spline-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--winner-tolerance",
        "--tolerance",
        dest="winner_tolerance",
        type=float,
        default=1e-9,
        help=(
            "Absolute tolerance for calling moderate-score differences ties "
            "(default: %(default)g)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = compare_metrics(
            qut_metrics=args.qut_metrics,
            spline_metrics=args.spline_metrics,
            output_dir=args.output_dir,
            winner_tolerance=args.winner_tolerance,
        )
    except ComparisonInputError as exc:
        parser.exit(2, f"error: {exc}\n")

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
