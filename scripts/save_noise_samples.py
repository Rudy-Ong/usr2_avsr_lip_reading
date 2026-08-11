from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchaudio

from data.transforms import AddNoise

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "csv" / "lrs2_test_1241.csv"
DEFAULT_NOISE = ROOT / "data" / "noise" / "babble_noise.npy"
DEFAULT_OUT_ROOT = ROOT / "data" / "noise"
LEVELS = [30, 25, 20, 15, 10, 5, 0, -5, -10]
TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize noisy .wav samples for a handful of test utterances, "
        "for manual listening/inspection (separate from the in-memory noise sweep).",
    )
    parser.add_argument("--csv-in", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--noise-path", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-utterances", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def utterance_id(audio_path: str) -> str:
    return Path(audio_path).stem


def main() -> None:
    args = parse_args()
    # Seed the global `random` module (not a local Random instance) since AddNoise's
    # noise-crop selection also draws from it — this makes both utterance selection and
    # the exact noise crop reproducible across reruns with the same --seed.
    random.seed(args.seed)

    with open(args.csv_in, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chosen = random.sample(rows, args.num_utterances)
    print(f"Selected {len(chosen)} utterances (seed={args.seed}):")
    for row in chosen:
        print(f"  {utterance_id(row['audio_path'])}")

    noise = AddNoise(noise_path=str(args.noise_path))

    for level in LEVELS:
        level_dir = args.out_root / f"noise{level}dB"
        level_dir.mkdir(parents=True, exist_ok=True)
        noise.snr_target = level

        for row in chosen:
            audio_path = row["audio_path"]
            uid = utterance_id(audio_path)

            clean, sr = torchaudio.load(audio_path, normalize=True)
            clean = clean.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                clean = torchaudio.functional.resample(clean, sr, TARGET_SR)

            noisy = noise(clean)
            out_path = level_dir / f"{uid}_noise{level}dB.wav"
            torchaudio.save(str(out_path), noisy, TARGET_SR)

        print(f"Wrote {len(chosen)} files to {level_dir}")


if __name__ == "__main__":
    main()
