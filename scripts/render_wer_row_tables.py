from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOINED = ROOT / "csv" / "worst_wer_examples.json"
DEFAULT_OUT_DIR = ROOT / "assets"

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
HIGHLIGHT = "#eb6834"  # orange — flags the modality this table is about

MODALITY_TITLE = {
    "v": "Visual-only (V)",
    "a": "Audio-only (A)",
    "av": "Audio-Visual (AV)",
}
COL_ORDER = ["index", "true_text", "pred_v", "pred_a", "pred_av"]
COL_LABELS = ["Index", "Text Reference", "Visual-only (V)", "Audio-only (A)", "Audio-Visual (AV)"]
COL_TO_MODALITY = {"pred_v": "v", "pred_a": "a", "pred_av": "av"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a small comparison table (like the paper's WER_ROW "
        "figures) for each modality's worst-WER examples.",
    )
    parser.add_argument("--joined-json", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def render_one(modality: str, rows: list, out_path: Path) -> None:
    n_rows = len(rows) + 1  # + header
    fig, ax = plt.subplots(figsize=(11, 0.62 * n_rows + 0.3), dpi=200)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")

    cell_text = [[str(r[c]) for c in COL_ORDER] for r in rows]
    table = ax.table(
        cellText=cell_text, colLabels=COL_LABELS,
        cellLoc="left", colLoc="left", loc="center",
        colWidths=[0.06, 0.24, 0.23, 0.23, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.1)

    highlighted_col = {"v": "pred_v", "a": "pred_a", "av": "pred_av"}[modality]
    highlighted_idx = COL_ORDER.index(highlighted_col)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRIDLINE)
        cell.set_linewidth(1)
        cell.PAD = 0.02
        if row == 0:
            cell.set_facecolor("#efeee9")
            cell.set_text_props(color=INK_PRIMARY, fontweight="bold")
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK_SECONDARY)
            if col == highlighted_idx:
                cell.set_facecolor("#fdf0e9")
                cell.set_text_props(color=INK_PRIMARY, fontweight="bold")

    wer_key = f"wer_{modality}"
    worst_pct = 100 * max(r[wer_key] for r in rows)
    ax.set_title(
        f"{MODALITY_TITLE[modality]} — biggest WER error {worst_pct:.1f}% "
        f"(indices {', '.join(str(r['index']) for r in rows)})",
        color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    data = json.loads(args.joined_json.read_text(encoding="utf-8"))
    worst = data["worst"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    filenames = {"v": "WER_ROW_(V).png", "a": "WER_ROW_(A).png", "av": "WER_ROW_(AV).png"}
    for modality, rows in worst.items():
        render_one(modality, rows, args.out_dir / filenames[modality])


if __name__ == "__main__":
    main()
