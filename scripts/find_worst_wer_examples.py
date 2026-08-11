from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join the clean-condition (inf) per-modality report CSVs by "
        "video_path and find the worst-WER examples per modality.",
    )
    parser.add_argument("--report-v", type=Path, required=True)
    parser.add_argument("--report-a", type=Path, required=True)
    parser.add_argument("--report-av", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--out-json", type=Path, default=ROOT / "csv" / "worst_wer_examples.json")
    return parser.parse_args()


def load(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["video_path"]: row for row in csv.DictReader(f)}


def main() -> None:
    args = parse_args()
    v_rows = load(args.report_v)
    a_rows = load(args.report_a)
    av_rows = load(args.report_av)

    common = sorted(set(v_rows) & set(a_rows) & set(av_rows))
    print(f"v={len(v_rows)} a={len(a_rows)} av={len(av_rows)} common={len(common)}")

    joined = []
    for idx, video_path in enumerate(common, start=1):
        v, a, av = v_rows[video_path], a_rows[video_path], av_rows[video_path]
        assert v["true_text"] == a["true_text"] == av["true_text"], video_path
        joined.append({
            "index": idx,
            "video_path": video_path,
            "audio_path": a["audio_path"],
            "true_text": v["true_text"],
            "pred_v": v["prediction_text"], "wer_v": float(v["wer"]),
            "pred_a": a["prediction_text"], "wer_a": float(a["wer"]),
            "pred_av": av["prediction_text"], "wer_av": float(av["wer"]),
        })

    worst = {}
    for modality in ("v", "a", "av"):
        ranked = sorted(joined, key=lambda r: r[f"wer_{modality}"], reverse=True)
        worst[modality] = ranked[: args.top_n]
        print(f"\nWorst {args.top_n} for {modality}:")
        for r in worst[modality]:
            print(f"  idx={r['index']} wer_{modality}={100*r[f'wer_{modality}']:.1f}% "
                  f"ref={r['true_text']!r} pred={r[f'pred_{modality}']!r}")

    args.out_json.write_text(json.dumps({"joined": joined, "worst": worst}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
