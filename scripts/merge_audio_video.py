from __future__ import annotations

import argparse
import csv
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from shutil import which
"""
Merge paired video/audio files by filename stem and export address paths CSV.
Example CLI usage:
    python scripts/merge_audio_video.py \
        --audio-dir path/to/audio \  
        --video-dir path/to/video \  
        --output-dir path/to/merged_output \  
        --csv-output path/to/merged_data.csv \  
"""
AUDIO_DIR = "path/to/audio"
VIDEO_DIR = "path/to/video"
OUTPUT_DIR = "path/to/merged_output"
CSV_OUTPUT_PATH = "path/to/merged_data.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge paired video/audio files by filename stem and export merged paths CSV.",
    )
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--csv-output", type=Path, default=CSV_OUTPUT_PATH)
    parser.add_argument("--audio-ext", default="wav")
    parser.add_argument("--video-ext", default="mp4")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. Default is skip existing outputs.",
    )
    return parser.parse_args()


def normalize_ext(ext: str) -> str:
    return ext[1:] if ext.startswith(".") else ext


def build_jobs(
    audio_dir: Path,
    video_dir: Path,
    output_dir: Path,
    audio_ext: str,
    video_ext: str,
    overwrite: bool,
) -> tuple[list[tuple[Path, Path, Path, bool]], list[str], list[str], int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_pattern = f"*.{normalize_ext(audio_ext)}"
    video_pattern = f"*.{normalize_ext(video_ext)}"

    audio_map = {path.stem: path for path in audio_dir.glob(audio_pattern)}
    video_map = {path.stem: path for path in video_dir.glob(video_pattern)}

    paired_stems = sorted(set(audio_map) & set(video_map))
    missing_audio = sorted(set(video_map) - set(audio_map))
    missing_video = sorted(set(audio_map) - set(video_map))

    jobs: list[tuple[Path, Path, Path, bool]] = []
    skipped_existing = 0
    for stem in paired_stems:
        output_path = output_dir / f"{stem}.mp4"
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        jobs.append((video_map[stem], audio_map[stem], output_path, overwrite))

    return jobs, missing_audio, missing_video, skipped_existing


def merge_one(job: tuple[Path, Path, Path, bool]) -> dict[str, str | int | bool]:
    video_file, audio_file, output_file, overwrite = job

    if output_file.exists() and not overwrite:
        return {
            "success": True,
            "skipped": True,
            "output": str(output_file),
            "stderr": "",
            "returncode": 0,
        }

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_file),
        "-i",
        str(audio_file),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-shortest",
        str(output_file),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return {
        "success": result.returncode == 0,
        "skipped": False,
        "output": str(output_file),
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def to_root_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(csv_path: Path, rows: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_path"])
        for value in rows:
            writer.writerow([value])


def check_ffmpeg_installed() -> bool:
    return which("ffmpeg") is not None


def main() -> None:
    args = parse_args()

    if not check_ffmpeg_installed():
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")

    jobs, missing_audio, missing_video, skipped_existing = build_jobs(
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        audio_ext=args.audio_ext,
        video_ext=args.video_ext,
        overwrite=args.overwrite,
    )

    paired_total = len(jobs) + skipped_existing
    print(f"Paired stems found: {paired_total}")
    print(f"Missing audio count: {len(missing_audio)}")
    print(f"Missing video count: {len(missing_video)}")
    if skipped_existing:
        print(f"Skipped existing outputs: {skipped_existing}")

    if not jobs and paired_total == 0:
        print("No matching audio/video pairs found.")
        write_csv(args.csv_output, [])
        print(f"CSV written: {args.csv_output}")
        return

    worker_count = args.workers if args.workers > 0 else min(os.cpu_count() or 1, max(len(jobs), 1))
    print(f"Merge jobs to run: {len(jobs)} (workers={worker_count})")

    success_count = 0
    fail_count = 0
    csv_rows: list[str] = []

    if jobs:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(merge_one, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                output_path = Path(str(result["output"]))

                if bool(result["success"]):
                    success_count += 1
                    print(f"[OK] {output_path}")
                    csv_rows.append(to_root_relative(output_path, ROOT))
                else:
                    fail_count += 1
                    print(f"[FAIL] {output_path}")
                    print(str(result["stderr"]).strip())

    if skipped_existing:
        for output_path in sorted(args.output_dir.glob("*.mp4")):
            csv_rows.append(to_root_relative(output_path, ROOT))

    csv_rows = sorted(set(csv_rows))
    write_csv(args.csv_output, csv_rows)

    print(f"Done. Success: {success_count}, Failed: {fail_count}")
    print(f"CSV written: {args.csv_output} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()