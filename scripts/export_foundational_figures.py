"""Export significant saved figures from the foundational single-game notebooks."""

from __future__ import annotations

import base64
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "foundational-single-game"
GAME_ID = "0021500009"


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def stream_text(output: dict) -> str:
    text = output.get("text", "")
    return "".join(text) if isinstance(text, list) else str(text)


def png_bytes(output: dict) -> bytes | None:
    encoded = output.get("data", {}).get("image/png")
    if encoded is None:
        return None
    if isinstance(encoded, list):
        encoded = "".join(encoded)
    return base64.b64decode(encoded)


def matching_code_cells(notebook: dict, marker: str) -> list[dict]:
    return [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and marker in source_text(cell)
    ]


def first_png(cell: dict) -> bytes:
    for output in cell.get("outputs", []):
        image = png_bytes(output)
        if image is not None:
            return image
    raise RuntimeError("The selected notebook cell has no saved PNG output.")


def select_figure(
    notebook: dict,
    source_marker: str,
    *,
    use_last_match: bool = False,
) -> bytes:
    cells = matching_code_cells(notebook, source_marker)
    if not cells:
        raise RuntimeError(f"No code cell contains marker: {source_marker!r}")
    return first_png(cells[-1] if use_last_match else cells[0])


def quarter_pointwise_figures(notebook: dict) -> list[bytes]:
    cells = matching_code_cells(notebook, "analyze_tv_logistic_per_period(df_original)")
    if len(cells) != 1:
        raise RuntimeError("Could not uniquely identify the saved per-quarter QUT cell.")

    figures = []
    awaiting_figure = False
    for output in cells[0].get("outputs", []):
        if "Plotting pointwise percentile maps" in stream_text(output):
            awaiting_figure = True
            continue
        if awaiting_figure:
            image = png_bytes(output)
            if image is not None:
                figures.append(image)
                awaiting_figure = False

    if len(figures) != 4:
        raise RuntimeError(
            f"Expected four saved quarter pointwise figures; found {len(figures)}."
        )
    return figures


def write_figure(filename: str, image: bytes) -> None:
    destination = OUTPUT_DIRECTORY / filename
    destination.write_bytes(image)
    print(destination.relative_to(PROJECT_ROOT))


def main() -> None:
    nba_notebook = json.loads(
        (PROJECT_ROOT / "NBA_Analysis.ipynb").read_text(encoding="utf-8")
    )
    qut_notebook = json.loads(
        (PROJECT_ROOT / "QUT_Boostrap.ipynb").read_text(encoding="utf-8")
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    nba_exports = (
        (
            "Inverted Ball Movement Heatmap on Non-Scoring Plays",
            "movement-heatmaps__non-scoring-possessions__quarters-1-to-4.png",
            False,
        ),
        (
            "Inverted Ball Movement Heatmap on Scoring Plays",
            "movement-heatmaps__scoring-possessions__quarters-1-to-4.png",
            False,
        ),
        (
            "Scoring Probability Maps by Period",
            "conditional-scoring-probability__minimum-sector-visits-3__"
            "quarters-1-to-4.png",
            True,
        ),
    )
    for marker, suffix, use_last_match in nba_exports:
        write_figure(
            f"game-{GAME_ID}__{suffix}",
            select_figure(
                nba_notebook,
                marker,
                use_last_match=use_last_match,
            ),
        )

    qut_exports = (
        (
            "boot_datasets = generate_bootstrapped_datasets(",
            "qut-lambda-distributions__full-game__bootstraps-500.png",
        ),
        (
            "theta_hat, theta_lo, theta_hi, maps = "
            "bootstrap_percentile_maps_fixed_lambda(",
            "qut-pointwise-coefficient-intervals__full-game__bootstraps-500.png",
        ),
    )
    for marker, suffix in qut_exports:
        write_figure(
            f"game-{GAME_ID}__{suffix}",
            select_figure(qut_notebook, marker),
        )

    for quarter, image in enumerate(quarter_pointwise_figures(qut_notebook), start=1):
        write_figure(
            f"game-{GAME_ID}__qut-pointwise-coefficient-intervals__"
            f"quarter-{quarter}__bootstraps-250.png",
            image,
        )


if __name__ == "__main__":
    main()
