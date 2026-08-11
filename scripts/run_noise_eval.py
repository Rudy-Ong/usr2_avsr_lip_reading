"""
USR 2.0 — Noise-robustness sweep driver.

Standalone script version of the row-by-row CSV evaluation loop used by
`inference_test_data.ipynb` (see README_inference_test_data.md), extended to:
  - sweep multiple modalities (a, v, av) and multiple SNR levels (dB) in one run,
  - reuse the loaded model/beam-search across every row and SNR level instead of
    reloading the checkpoint per call (unlike utils.inference_.transcribe),
  - reuse each row's mouth-crop video tensor across modalities/SNR levels, since
    noise only ever perturbs the audio branch.

Usage (from repo root):
    .venv/bin/python scripts/run_noise_eval.py \\
        modality=a,v,av max_rows=10 snr_levels=inf,10,0,-10 \\
        decode.beam_size=1 decode.ctc_weight=0.0

`snr_levels` accepts integers and the special label "inf" (case-insensitive), which maps
to AddNoise's built-in no-noise sentinel (snr_target=9999) while still being reported as
"inf" in filenames/summary rows.

New (non-preexisting) Hydra keys: snr_levels, noise_path, out_dir (see conf/config.yaml).
To override a key that isn't already declared in conf/, prefix with '+', e.g. `+foo=bar`.
"""

import csv
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from data.transforms import AddNoise
from metrics import WER, get_wer
from preprocessing.landmarks_detector import LandmarksDetector
from preprocessing.video_preprocess import VideoProcess
from utils.inference_ import (
    build_beam_search,
    decode,
    load_audio_file,
    load_model,
    load_video_audio,
    preprocess_mouth_roi_video,
    preprocess_video,
    resolve_inference_video_path,
)

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("len", len, replace=True)

REPORT_FIELDS = ["dataset", "video_path", "audio_path", "true_text", "status", "prediction_text", "wer"]


NO_NOISE_SENTINEL = 9999


def parse_snr_levels(raw):
    """Returns a list of (label, numeric) pairs. label is the human-readable string used
    in filenames/summary rows ("inf" or the integer as text); numeric is what's actually
    fed to AddNoise (9999 for "inf", matching its built-in no-noise sentinel)."""
    levels = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "inf":
            levels.append(("inf", NO_NOISE_SENTINEL))
        else:
            levels.append((token, int(token)))
    return levels


def read_test_rows(csv_path, max_rows):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows > 0 and len(rows) >= max_rows:
                break
            rows.append(row)
    return rows


def build_video_tensor(video_path, cfg, ld, vp, mouth_roi_root):
    """Mirrors utils.inference_.transcribe's v/av video-loading branch."""
    resolved_path = resolve_inference_video_path(video_path, cfg, "av")
    if resolved_path != video_path:
        video_frames, _ = load_video_audio(resolved_path)
        return preprocess_mouth_roi_video(video_frames)
    video_frames, _ = load_video_audio(video_path)
    return preprocess_video(video_frames, ld, vp, video_path, mouth_roi_root=mouth_roi_root)


def align_audio_to_video(audio_t, video_tensor, cfg):
    """Mirrors utils.inference_.transcribe's AV alignment (pad/trim audio to video length)."""
    video_frames_count = int(video_tensor.shape[0])
    samples_per_frame = int(cfg.get("audio_samples_per_video_frame", 16000 // 25))
    target_samples = max(video_frames_count * samples_per_frame, 1)
    current_samples = int(audio_t.shape[1])
    if current_samples < target_samples:
        audio_t = torch.nn.functional.pad(audio_t, (0, target_samples - current_samples))
    elif current_samples > target_samples:
        audio_t = audio_t[:, :target_samples]
    return audio_t


@torch.no_grad()
def run_one(modality, video_tensor, video_error, audio_path, snr, noise, device, model, beam_search, cfg):
    if modality in ("v", "av") and video_error is not None:
        raise video_error

    audio_t = None
    if modality in ("a", "av"):
        audio_t = load_audio_file(audio_path)
        # AddNoise's snr_target==9999 branch returns a differently-shaped (1D) tensor
        # than its normal (2D) branch — dormant in the original pipeline since noise_path
        # defaults to None there. Skip the call entirely for "no noise" instead.
        if snr is not None and snr != NO_NOISE_SENTINEL:
            noise.snr_target = snr
            audio_t = noise(audio_t)
        if modality == "av":
            audio_t = align_audio_to_video(audio_t, video_tensor, cfg)

    if modality == "av":
        audio_input = audio_t.unsqueeze(0).to(device).transpose(1, 2)
        video_input = video_tensor.unsqueeze(0).to(device)
        feat = model.encoder(xs_v=video_input, xs_a=audio_input)
    elif modality == "v":
        video_input = video_tensor.unsqueeze(0).to(device)
        feat = model.encoder(xs_v=video_input)
    else:
        audio_input = audio_t.unsqueeze(0).to(device).transpose(1, 2)
        feat = model.encoder(xs_a=audio_input)

    return decode(feat, beam_search, modality, cfg)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    modalities = [m.strip() for m in str(cfg.get("modality", "av")).split(",") if m.strip()]
    for m in modalities:
        if m not in ("a", "v", "av"):
            print(f"Error: invalid modality '{m}'. Use a, v, and/or av (comma-separated).")
            sys.exit(1)

    snr_levels = parse_snr_levels(cfg.get("snr_levels", "0,5,10"))
    csv_path = cfg.get("csv_path") or "csv/lrs2_test_modified.csv"
    max_rows = int(cfg.get("max_rows", -1))
    noise_path = cfg.get("noise_path", "data/noise/babble_noise.npy")
    out_dir = cfg.get("out_dir", "csv")
    detector = cfg.get("detector", "mediapipe")
    mouth_roi_root = cfg.get("mouth_roi_root", "data/mouth_roi")
    seed = int(cfg.get("seed", 42))

    # AddNoise (data/transforms.py) picks its noise crop via the global `random` module,
    # unseeded by default — at high SNR the injected noise is tiny, so an unseeded crop
    # makes a handful of borderline utterances flip unpredictably between runs/levels,
    # which reads as noise "improving" WER between adjacent high-SNR points. Fixing the
    # seed once here makes every run reproducible (the same code + seed always reproduces
    # the exact same grid).
    random.seed(seed)
    print(f"Seed: {seed}")

    needs_video = any(m in ("v", "av") for m in modalities)
    needs_audio = any(m in ("a", "av") for m in modalities)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.disable(logging.CRITICAL)  # keep terminal output minimal during the batch sweep

    log.info("Loading model from: %s", cfg.model.pretrained_model_path)
    model = load_model(cfg, cfg.model.pretrained_model_path, device)
    beam_search = build_beam_search(cfg, model)
    beam_search.to(device)

    noise = AddNoise(noise_path=noise_path, snr_target=snr_levels[0][1]) if needs_audio else None

    ld = vp = None
    if needs_video:
        ld = LandmarksDetector(detector=detector)
        vp = VideoProcess(convert_gray=False)

    rows = read_test_rows(csv_path, max_rows)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    print(f"Modalities: {modalities}  SNR levels: {[label for label, _ in snr_levels]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    # results[(modality, snr_label)] -> list of report-row dicts
    results = {(m, label): [] for m in modalities for label, _ in snr_levels}

    try:
        for row in rows:
            video_path = row.get("video_path", "").strip()
            audio_path = row.get("audio_path", "").strip()
            true_text = (row.get("true_text") or "").strip()

            video_tensor, video_error = None, None
            if needs_video:
                try:
                    video_tensor = build_video_tensor(video_path, cfg, ld, vp, mouth_roi_root)
                except Exception as exc:  # noqa: BLE001 - recorded per-row, matches evaluate_from_csv convention
                    video_error = exc

            def decode_row(modality, snr_numeric):
                try:
                    pred_text = run_one(
                        modality, video_tensor, video_error, audio_path, snr_numeric,
                        noise, device, model, beam_search, cfg,
                    )
                    wer = get_wer(pred_text, true_text) if true_text else None
                    status = "ok"
                except Exception as exc:  # noqa: BLE001 - matches evaluate_from_csv convention
                    pred_text = f"<error: {type(exc).__name__}: {exc}>"
                    wer = None
                    status = "error"
                return {
                    "dataset": row.get("dataset", ""),
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "true_text": true_text,
                    "status": status,
                    "prediction_text": pred_text,
                    "wer": wer,
                }

            for modality in modalities:
                if modality == "v":
                    # v never sees the audio/noise branch, so its output can't vary with
                    # SNR — decode once per row and replicate across every SNR label
                    # instead of paying for a redundant beam search per level.
                    row_result = decode_row(modality, None)
                    for label, _ in snr_levels:
                        results[(modality, label)].append(dict(row_result))
                else:
                    for label, numeric in snr_levels:
                        results[(modality, label)].append(decode_row(modality, numeric))
    finally:
        if ld is not None:
            ld.close()

    logging.disable(logging.NOTSET)

    summary_path = os.path.join(out_dir, "noise_sweep_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", "snr_db", "num_ok", "num_error", "corpus_wer"])

        for modality in modalities:
            for label, _ in snr_levels:
                report_rows = results[(modality, label)]
                level_tag = "clean" if label == "inf" else f"{label}dB"
                report_path = os.path.join(out_dir, f"[report]{modality}_noise_{level_tag}_{timestamp}.csv")
                with open(report_path, "w", newline="", encoding="utf-8") as rf:
                    rwriter = csv.DictWriter(rf, fieldnames=REPORT_FIELDS)
                    rwriter.writeheader()
                    rwriter.writerows(report_rows)

                wer_metric = WER()
                num_ok = num_error = 0
                for r in report_rows:
                    if r["status"] == "ok" and r["true_text"]:
                        wer_metric.update(r["prediction_text"], r["true_text"])
                        num_ok += 1
                    else:
                        num_error += 1
                corpus_wer = wer_metric.compute().item() if num_ok > 0 else float("nan")

                writer.writerow([modality, label, num_ok, num_error, f"{corpus_wer:.6f}"])
                print(f"modality={modality:<2} snr={label:>4}dB  ok={num_ok:<5} error={num_error:<4} "
                      f"corpus_wer={corpus_wer:.4f}  -> {report_path}")

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
