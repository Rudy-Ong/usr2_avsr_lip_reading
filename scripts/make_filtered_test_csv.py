from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "csv" / "lrs2_test_modified.csv"
DEFAULT_OUT = ROOT / "csv" / "lrs2_test_1241.csv"

# Known face-detection failures (confirmed against the historical visual/av inference
# reports: RuntimeError "Could not detect a face in enough frames"). Audio-only decode
# works fine for both, so filtering them out here gives a/v/av one shared, fully-usable
# datapath instead of relying on per-row error handling at eval time.
KNOWN_BAD_VIDEO_PATHS = {
    "data/lrs2/main/6351431997518095202/00017.mp4",
    "data/lrs2/main/6375448166147161013/00031.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the known face-detection-failure utterances out of the test "
        "manifest, producing one shared datapath usable for a/v/av evaluation.",
    )
    parser.add_argument("--in-csv", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.in_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = [r for r in rows if r["video_path"].strip() not in KNOWN_BAD_VIDEO_PATHS]
    dropped = len(rows) - len(kept)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Read {len(rows)} rows from {args.in_csv}")
    print(f"Dropped {dropped} known-bad rows: {sorted(KNOWN_BAD_VIDEO_PATHS)}")
    print(f"Wrote {len(kept)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
