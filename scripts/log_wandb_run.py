from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOINED = ROOT / "csv" / "worst_wer_examples.json"
DEFAULT_SUMMARY = ROOT / "csv" / "noise_sweep_summary.csv"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log the clean-condition (inf) LRS2 eval run to wandb: full "
        "per-utterance predictions table + worst-case examples + summary WER.",
    )
    parser.add_argument("--joined-json", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-name", default="lrs2_1241_clean_inf")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT / ".env")
    import wandb

    args = parse_args()
    data = json.loads(args.joined_json.read_text(encoding="utf-8"))
    joined, worst = data["joined"], data["worst"]

    import csv
    corpus_wer = {}
    with open(args.summary_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["snr_db"] == "inf":
                corpus_wer[row["modality"]] = float(row["corpus_wer"])

    run = wandb.init(
        project=os.environ.get("WANDB_project", "usr2.0"),
        job_type="eval",
        name=args.run_name,
        config={
            "checkpoint": "huge_high_resource_lrs2lrs3vox2avsp",
            "backbone": "resnet_transformer_huge",
            "dataset": "lrs2_test_1241",
            "num_utterances": len(joined),
            "condition": "clean (inf, no noise)",
            "decode.beam_size": 30,
            "decode.ctc_weight": 0.1,
            "decode.maxlenratio": 0.4,
        },
    )

    wandb.summary["wer_visual"] = corpus_wer.get("v")
    wandb.summary["wer_audio"] = corpus_wer.get("a")
    wandb.summary["wer_audio_visual"] = corpus_wer.get("av")

    full_table = wandb.Table(
        columns=["index", "video_path", "true_text", "pred_v", "wer_v", "pred_a", "wer_a", "pred_av", "wer_av"],
        data=[[r["index"], r["video_path"], r["true_text"],
               r["pred_v"], r["wer_v"], r["pred_a"], r["wer_a"], r["pred_av"], r["wer_av"]]
              for r in joined],
    )
    wandb.log({"predictions/all_utterances": full_table})

    for modality, rows in worst.items():
        wer_key = f"wer_{modality}"
        worst_table = wandb.Table(
            columns=["index", "true_text", "pred_v", "pred_a", "pred_av", wer_key],
            data=[[r["index"], r["true_text"], r["pred_v"], r["pred_a"], r["pred_av"], r[wer_key]]
                  for r in rows],
        )
        wandb.log({f"predictions/worst_{modality}": worst_table})

    print(f"Logged run: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
