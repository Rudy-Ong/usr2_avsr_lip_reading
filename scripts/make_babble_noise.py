from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "audio"
OUTPUT_PATH = ROOT / "data" / "noise" / "babble_noise.npy"
TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local babble-noise track by overlap-summing random speech "
        "clips from the test set's own audio pool (stand-in for the paper's babble_noise.npy).",
    )
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--num-speakers", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = torchaudio.load(str(path), normalize=True)
    audio = audio.mean(dim=0)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
    return audio.numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    wav_files = sorted(args.audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under {args.audio_dir}")

    num_speakers = min(args.num_speakers, len(wav_files))
    chosen = rng.sample(wav_files, num_speakers)

    target_len = int(args.duration_sec * TARGET_SR)
    babble = np.zeros(target_len, dtype=np.float32)

    for path in chosen:
        clip = load_mono_16k(path)
        if len(clip) == 0:
            continue
        # Tile the clip to cover the full track, then place it at a random offset
        # so speakers don't all start in phase.
        reps = target_len // len(clip) + 2
        looped = np.tile(clip, reps)
        offset = rng.randint(0, len(clip) - 1)
        babble += looped[offset:offset + target_len]

    babble /= num_speakers

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, babble)
    print(f"Wrote {args.out} ({len(babble)} samples, {args.duration_sec:.1f}s @ {TARGET_SR}Hz, "
          f"built from {num_speakers} speakers)")


if __name__ == "__main__":
    main()
