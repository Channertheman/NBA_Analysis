"""Export the saved QUT method-comparison figures from projections.ipynb."""

from __future__ import annotations

import base64
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "projections.ipynb"
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "qut-method-comparison"
GAME_IDS = ("0021500009", "0021500390")
FIGURE_HEADING = "Sector coefficients across QUT methods"


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def output_text(output: dict) -> str:
    data = output.get("data", {})
    markdown = data.get("text/markdown", "")
    if isinstance(markdown, list):
        return "".join(markdown)
    return str(markdown)


def png_bytes(output: dict) -> bytes | None:
    encoded = output.get("data", {}).get("image/png")
    if encoded is None:
        return None
    if isinstance(encoded, list):
        encoded = "".join(encoded)
    return base64.b64decode(encoded)


def find_comparison_figure(notebook: dict, game_id: str) -> bytes:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = source_text(cell)
        if f'game_id="{game_id}"' not in source:
            continue

        comparison_heading_seen = False
        for output in cell.get("outputs", []):
            if FIGURE_HEADING in output_text(output):
                comparison_heading_seen = True
                continue
            if comparison_heading_seen:
                image = png_bytes(output)
                if image is not None:
                    return image

    raise RuntimeError(
        f"Could not find the saved method-comparison figure for game {game_id}."
    )


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for game_id in GAME_IDS:
        destination = OUTPUT_DIRECTORY / (
            f"game-{game_id}__full-and-reduced-grid__"
            "bounded-slack-vs-moore-penrose-projection.png"
        )
        destination.write_bytes(find_comparison_figure(notebook, game_id))
        print(destination.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
