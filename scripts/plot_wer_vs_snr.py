from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "csv" / "noise_sweep_summary.csv"
DEFAULT_OUT = ROOT / "wer_vs_snr.png"

# X order matches scripts/run_noise_eval.py's snr_levels schedule: no-noise sentinel
# first, then descending SNR (cleaner -> noisier).
X_LABELS = ["inf", "30", "25", "20", "15", "10", "5", "0", "-5", "-10"]

# dataviz skill reference palette (references/palette.md), light-mode categorical
# slots 1/2/3 (blue/orange/aqua) — chosen because that specific triple is the one
# documented as passing the ALL-PAIRS gate (not just adjacent), the stricter bar a
# 3-series chart needs: worst pair CVD dE 9.2 light (>=8 target), normal-vision dE
# 24.0 light (>=15 floor). `node scripts/validate_palette.js` (in the dataviz skill)
# would normally re-verify this, but node isn't available in this environment, so
# this relies on the skill's own published measurement for this exact triple.
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = {
    "a": {"label": "Audio (ASR)", "color": "#2a78d6", "marker": "o"},
    "v": {"label": "Visual (VSR)", "color": "#eb6834", "marker": "^"},
    "av": {"label": "Audio-Visual (AVSR)", "color": "#1baf7a", "marker": "s"},
}
ORDER = ["a", "v", "av"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the noise-sweep summary CSV as a WER-vs-SNR line chart, "
        "one line per modality.",
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_data(summary_csv: Path):
    rows = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["modality"], row["snr_db"])] = float(row["corpus_wer"]) * 100
    return {key: [rows[(key, label)] for label in X_LABELS] for key in ORDER}


def main() -> None:
    args = parse_args()
    data = load_data(args.summary_csv)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(SURFACE)

    x = list(range(len(X_LABELS)))

    for key in ORDER:
        spec = SERIES[key]
        ax.plot(
            x, data[key],
            color=spec["color"], linewidth=2, solid_capstyle="round",
            marker=spec["marker"], markersize=8,
            markerfacecolor=spec["color"], markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=spec["label"], zorder=3,
        )

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} dB" if lab != "inf" else "inf" for lab in X_LABELS],
                        color=INK_MUTED, fontsize=10)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], color=INK_MUTED, fontsize=10)
    ax.set_ylim(-4, 112)
    ax.set_xlim(-0.4, len(X_LABELS) - 0.6)

    ax.set_xlabel("SNR  (higher = cleaner; \"inf\" = no noise added)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("Word Error Rate (%)", color=INK_SECONDARY, fontsize=11)
    ax.set_title("USR 2.0 Noise Robustness — WER vs SNR by Modality",
                 color=INK_PRIMARY, fontsize=14, fontweight="bold", loc="left", pad=20)
    ax.text(0, 1.02, "1241 test utterances · beam_size=30, ctc_weight=0.1 · babble noise, seed=42",
            transform=ax.transAxes, color=INK_MUTED, fontsize=9, va="bottom")

    ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.98), frameon=False,
              fontsize=10, labelcolor=INK_PRIMARY, handletextpad=0.6)

    # Data label at every point (per explicit request). Audio and Audio-Visual run
    # almost coincident for long stretches (e.g. both ~1.3-1.9% from inf to 15dB),
    # so a fixed offset isn't enough — greedily push overlapping labels apart in
    # data space (sorted ascending, each forced >= MIN_SEP above the previous one)
    # and draw a thin leader line back to the true point wherever a label had to
    # move, so displaced labels stay traceable to their mark.
    MIN_SEP = 5.5  # percentage points between adjacent label centers
    for i in x:
        triples = sorted(((key, data[key][i]) for key in ORDER), key=lambda t: t[1])
        text_y = [val for _, val in triples]
        for j in range(1, len(text_y)):
            text_y[j] = max(text_y[j], text_y[j - 1] + MIN_SEP)

        for (key, val), ty in zip(triples, text_y):
            if abs(ty - val) > 1.0:
                ax.plot([i, i], [val, ty], color=INK_MUTED, linewidth=0.75,
                        alpha=0.6, zorder=2, solid_capstyle="butt")
            ax.text(i, ty, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=INK_SECONDARY, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                               edgecolor="none", alpha=0.85))

    fig.tight_layout()
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
