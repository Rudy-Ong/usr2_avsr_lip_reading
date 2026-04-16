"""
USR 2.0 — Batch-video inference demo.
"""

import logging
import csv
import os
import sys
from typing import Any, Dict, Optional, cast
import cv2
import hydra
import numpy as np
import torch
import torchaudio
try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    VideoDecoder = None
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import CenterCrop, Compose, Grayscale, Lambda, Normalize
from espnet.asr.asr_utils import parse_hypothesis
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from preprocessing.landmarks_detector import LandmarksDetector
from preprocessing.video_preprocess import VideoProcess
from metrics import WER
from utils.utils import UNIGRAM1000_LIST
from datetime import datetime
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video / audio loading
# ---------------------------------------------------------------------------

def load_video_audio(path: str, target_fps: int = 25, target_sr: int = 16000):
    """Load a video file and return (video_frames, audio_waveform).

    Returns
    -------
    video : np.ndarray, uint8, (T, H, W, C) RGB
    audio : torch.Tensor, float32, (1, S)  mono waveform at ``target_sr``
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    def _decode_with_torchcodec(video_path: str):
        """Best-effort TorchCodec decode. Returns (video_np, fps) or (None, None)."""
        if VideoDecoder is None:
            return None, None

        try:
            decoder = VideoDecoder(video_path)

            frame_batch = None
            if hasattr(decoder, "get_all_frames"):
                frame_batch = decoder.get_all_frames()
            elif hasattr(decoder, "__getitem__"):
                frame_batch = decoder[:]

            if frame_batch is None:
                return None, None

            if hasattr(frame_batch, "data"):
                data = frame_batch.data
            else:
                data = frame_batch

            if not torch.is_tensor(data):
                return None, None

            if data.ndim == 4 and data.shape[-1] in (1, 3):
                # Expected (T, H, W, C)
                video_np = data.cpu().numpy()
            elif data.ndim == 4 and data.shape[1] in (1, 3):
                # Handle (T, C, H, W)
                video_np = data.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            else:
                return None, None

            fps = None
            if hasattr(decoder, "metadata") and decoder.metadata is not None:
                fps = getattr(decoder.metadata, "average_fps", None)
                if fps is None:
                    fps = getattr(decoder.metadata, "fps", None)

            return video_np, fps
        except Exception as exc:
            log.warning("TorchCodec decode failed for %s: %s", video_path, exc)
            return None, None

    # --- video ---------------------------------------------------------------
    # Prefer TorchCodec when available, fallback to OpenCV.
    video, vfps = _decode_with_torchcodec(path)
    if video is None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")

        vfps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise RuntimeError(f"No decodable frames found in video: {path}")

        video = np.stack(frames, axis=0)

    if not vfps or vfps <= 1e-3:
        log.warning(
            "Could not determine video FPS from metadata. Assuming %d FPS. "
            "If your video has a different frame rate, results may be degraded. "
            "Consider re-encoding: ffmpeg -i input.mp4 -r 25 -ar 16000 output.mp4",
            target_fps
        )
        vfps = target_fps
    if abs(vfps - target_fps) > 1e-3:
        n_frames = len(video)
        new_n = int(n_frames / vfps * target_fps)
        new_n = max(new_n, 1)
        indices = torch.linspace(0, n_frames - 1, new_n).long().numpy()
        video = video[indices]

    # --- audio ---------------------------------------------------------------
    # OpenCV does not decode audio. Try sidecar WAV with the same basename.
    wav_path = os.path.splitext(path)[0] + ".wav"
    if os.path.exists(wav_path):
        audio, sr = torchaudio.load(wav_path, normalize=True)
        if audio.numel() == 0:
            audio = torch.zeros(1, 1)
            sr = target_sr
        else:
            audio = audio.mean(dim=0, keepdim=True)  # stereo -> mono
        if int(sr) != target_sr:
            audio = torchaudio.transforms.Resample(int(sr), target_sr)(audio)
    else:
        # Synthesise silence matching video duration when no sidecar audio exists.
        n_samples = int(video.shape[0] / target_fps * target_sr)
        audio = torch.zeros(1, max(n_samples, 1))

    # align audio length to video
    expected_samples = video.shape[0] * (target_sr // target_fps)
    if audio.shape[1] < expected_samples:
        audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.shape[1]))
    else:
        audio = audio[:, :expected_samples]

    return video, audio


def load_audio_file(path: str, target_sr: int = 16000):
    """Load an audio file and return mono waveform (1, S) at target_sr."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = torchaudio.load(path, normalize=True)
    if audio.numel() == 0:
        return torch.zeros(1, 1)

    audio = audio.mean(dim=0, keepdim=True)
    if int(sr) != target_sr:
        audio = torchaudio.transforms.Resample(int(sr), target_sr)(audio)
    return audio


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_video_transform():
    """Mouth-crop -> normalised grayscale tensor (T, H, W)."""
    return Compose([
        Lambda(lambda x: x / 255.0),
        CenterCrop(88),
        Lambda(lambda x: x.transpose(0, 1)),  # (C,T,H,W) -> (T,C,H,W) for Grayscale
        Grayscale(),
        Lambda(lambda x: x.transpose(0, 1)),  # (T,1,H,W) -> (1,T,H,W) for Normalize
        Normalize(mean=(0.421,), std=(0.165,)),
        Lambda(lambda x: x.squeeze(0)),       # (1,T,H,W) -> (T,H,W)

    ])


def save_mouth_crop(mouth_video: np.ndarray, output_path: str = "mouth_crop.mp4", fps: int = 25):
    """Save the raw mouth crop as a video file for inspection."""
    T, H, W, C = mouth_video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for t in range(T):
        frame = cv2.cvtColor(mouth_video[t], cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    log.info("Saved mouth crop to: %s (%d frames, %dx%d)", output_path, T, W, H)


def build_mouth_roi_output_path(video_path: str, output_root: str = "data/mouth_roi") -> str:
    """Create output path: output_root/parentFolder_filename.mp4."""
    video_path = os.path.normpath(video_path)
    parent = os.path.basename(os.path.dirname(video_path))
    filename = os.path.basename(video_path)
    name, _ = os.path.splitext(filename)

    if parent:
        out_name = f"{parent}_{name}.mp4"
    else:
        out_name = f"{name}.mp4"

    os.makedirs(output_root, exist_ok=True)
    return os.path.join(output_root, out_name)


def resolve_inference_video_path(video_path: str, cfg: DictConfig, modality: str) -> str:
    """For v/av, check if mouth ROI exists at the new address and optionally use it."""
    modality_key = modality[0].lower() if modality not in ("a", "v", "av") else modality
    if modality_key not in ("v", "av"):
        return video_path

    mouth_roi_root = cfg.get("mouth_roi_root", "data/mouth_roi")
    use_mouth_roi_input = bool(cfg.get("use_mouth_roi_input", False))
    roi_path = build_mouth_roi_output_path(video_path, output_root=mouth_roi_root)

    if os.path.exists(roi_path):
        log.info("Found mouth ROI at new path: %s", roi_path)
        if use_mouth_roi_input:
            log.info("Using mouth ROI as inference input for modality=%s", modality_key)
            return roi_path
    else:
        log.warning("Mouth ROI not found at new path: %s", roi_path)

    return video_path


def preprocess_mouth_roi_video(video_frames: np.ndarray):
    """Preprocess already-cropped mouth ROI video into model input tensor."""
    video_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2).float()
    video_tensor = build_video_transform()(video_tensor)
    return video_tensor


def preprocess_video(
    video_frames: np.ndarray,
    landmarks_detector,
    video_processor,
    source_video_path: str,
    mouth_roi_root: str = "data/mouth_roi",
):
    """Detect landmarks, crop mouth, return tensor (C, T, H, W)."""
    landmarks = landmarks_detector(video_frames)
    mouth_video = video_processor(video_frames, landmarks)
    if mouth_video is None:
        raise RuntimeError(
            "Could not detect a face in enough frames. "
            "Make sure the video contains a clearly visible face."
        )
    roi_output_path = build_mouth_roi_output_path(source_video_path, output_root=mouth_roi_root)
    save_mouth_crop(mouth_video, output_path=roi_output_path)
    # (T, H, W, C) -> (C, T, H, W) float tensor
    video_tensor = torch.from_numpy(mouth_video).permute(3, 0, 1, 2).float()
    video_tensor = build_video_transform()(video_tensor)
    return video_tensor  # (T, H, W)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: DictConfig, checkpoint_path: str, device: torch.device):
    """Instantiate E2E model, load weights, set to eval."""
    model = E2E(len(UNIGRAM1000_LIST), cfg.model.backbone)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Strip torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}
    # Strip wrapper module prefix if present
    if any(k.startswith("model.backbone.") for k in ckpt):
        ckpt = {
            k.replace("model.backbone.", "", 1): v
            for k, v in ckpt.items()
            if k.startswith("model.backbone.")
        }
    model.load_state_dict(ckpt)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def build_beam_search(cfg: DictConfig, model: E2E):
    """Build a BatchBeamSearch scorer."""
    token_list = UNIGRAM1000_LIST
    scorers = cast(Dict[str, Any], model.scorers())
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        length_bonus=cfg.decode.penalty,
    )
    return BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=len(token_list) - 1,
        eos=len(token_list) - 1,
        token_list=token_list,
        pre_beam_score_key=cast(Any, None if cfg.decode.ctc_weight == 1.0 else "decoder"),
    )


def decode(features: torch.Tensor, beam_search: BatchBeamSearch,
           modality: str, cfg: DictConfig) -> str:
    """Run beam search and return the 1-best transcription string."""
    hyps = beam_search(
        x=features.squeeze(0),
        modality=modality,
        maxlenratio=cfg.decode.maxlenratio,
        minlenratio=cfg.decode.minlenratio,
    )
    best = hyps[0].asdict()
    text, _, _, _ = parse_hypothesis(best, UNIGRAM1000_LIST)
    text = text.replace("<eos>", "").replace("\u2581", " ").strip()
    return text

@torch.no_grad()
def transcribe(
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    cfg: Optional[DictConfig] = None,
    modality: str = "av",
    device: torch.device = torch.device("cuda"),
    detector: str = "mediapipe",
):
    if cfg is None and isinstance(audio_path, DictConfig):
        cfg = audio_path
        audio_path = None
    if cfg is None:
        raise ValueError("cfg is required.")

    modality_key = modality[0].lower() if modality not in ("a", "v", "av") else modality

    if modality_key == "av":
        if not video_path:
            raise ValueError("AV modality requires video_path.")
        if not audio_path:
            audio_path = video_path
            log.warning("AV modality: audio_path missing, fallback to video_path for audio decode.")
    elif modality_key == "a":
        if not audio_path:
            raise ValueError("A modality requires audio_path.")
    elif modality_key == "v":
        if not video_path:
            raise ValueError("V modality requires video_path.")
    else:
        raise ValueError(f"Unknown modality '{modality}'. Choose from: av, v, a.")

    audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma"}
    video_tensor = None
    audio = None
    required_video_path: Optional[str] = video_path
    required_audio_path: Optional[str] = audio_path

    if modality_key in ("v", "av"):
        if required_video_path is None:
            raise ValueError("video_path is required for this modality")

        resolved_video_path = resolve_inference_video_path(required_video_path, cfg, modality)
        log.info("Loading video input: %s", resolved_video_path)

        use_mouth_roi_input = bool(cfg.get("use_mouth_roi_input", modality_key == "av"))
        mouth_roi_root = cfg.get("mouth_roi_root", "data/mouth_roi")

        if use_mouth_roi_input and resolved_video_path != video_path:
            video_frames, _ = load_video_audio(resolved_video_path)
        else:
            video_frames, _ = load_video_audio(required_video_path)

        if use_mouth_roi_input and resolved_video_path != video_path:
            log.info("Using existing mouth ROI input; skipping landmark detection.")
            video_tensor = preprocess_mouth_roi_video(video_frames)
        else:
            log.info("Detecting landmarks and cropping mouth region ...")
            ld = LandmarksDetector(detector=detector)
            vp = VideoProcess(convert_gray=False)
            try:
                video_tensor = preprocess_video(
                    video_frames,
                    ld,
                    vp,
                    required_video_path,
                    mouth_roi_root=mouth_roi_root,
                )
            finally:
                ld.close()

    if modality_key in ("a", "av"):
        if required_audio_path is None:
            raise ValueError("audio_path is required for this modality")
        log.info("Loading audio input: %s", required_audio_path)
        audio_ext = os.path.splitext(required_audio_path)[1].lower()
        
        if audio_ext in audio_exts:
            audio = load_audio_file(required_audio_path)
        else:
            _, audio = load_video_audio(required_audio_path)

    if modality_key in ("v", "av") and video_tensor is None:
        raise RuntimeError("Failed to prepare video tensor for modality using video.")
    if modality_key in ("a", "av") and audio is None:
        raise RuntimeError("Failed to prepare audio tensor for modality using audio.")

    video_tensor_t = cast(torch.Tensor, video_tensor)
    audio_t = cast(torch.Tensor, audio)

    if modality_key == "av":
        video_frames_count = int(video_tensor_t.shape[0])
        samples_per_frame = int(cfg.get("audio_samples_per_video_frame", 16000 // 25))
        target_samples = max(video_frames_count * samples_per_frame, 1)
        current_samples = int(audio_t.shape[1])

        if current_samples < target_samples:
            audio_t = torch.nn.functional.pad(audio_t, (0, target_samples - current_samples))
            log.info(
                "AV alignment: audio padded from %d to %d samples (video frames=%d).",
                current_samples,
                target_samples,
                video_frames_count,
            )
        elif current_samples > target_samples:
            audio_t = audio_t[:, :target_samples]
            log.info(
                "AV alignment: audio trimmed from %d to %d samples (video frames=%d).",
                current_samples,
                target_samples,
                video_frames_count,
            )

    log.info("Loading model from: %s", cfg.model.pretrained_model_path)
    model = load_model(cfg, cfg.model.pretrained_model_path, device)
    beam_search = build_beam_search(cfg, model)
    beam_search.to(device)

    if modality_key == "av":
        audio_input = audio_t.unsqueeze(0).to(device).transpose(1, 2)
        video_input = video_tensor_t.unsqueeze(0).to(device)
        feat = model.encoder(xs_v=video_input, xs_a=audio_input)
    elif modality_key == "v":
        video_input = video_tensor_t.unsqueeze(0).to(device)
        feat = model.encoder(xs_v=video_input)
    else:
        audio_input = audio_t.unsqueeze(0).to(device).transpose(1, 2)
        feat = model.encoder(xs_a=audio_input)

    log.info("Decoding with modality=%s, beam_size=%d ...", modality_key, cfg.decode.beam_size)
    return decode(feat, beam_search, modality_key, cfg)

def to_txt_path(video_path: str) -> str:
    base, _ = os.path.splitext(video_path)
    return base + ".txt"


def read_first_line(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Empty reference text in: {path}")
    return line

def evaluate_from_csv(cfg: DictConfig, device: torch.device, modality: str, detector: str):
    csv_path = cfg.get("csv_path", "csv/lrs2_test_modified.csv")
    if not os.path.exists(csv_path):
        fallback_csv = "csv/lrs2_datapath.csv"
        if os.path.exists(fallback_csv):
            log.warning("CSV not found: %s. Falling back to %s", csv_path, fallback_csv)
            csv_path = fallback_csv
        else:
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}. Also missing fallback: {fallback_csv}"
            )

    lrs2_root = cfg.get("lrs2_root", "data/lrs2")
    max_rows = int(cfg.get("max_rows", -1))
    modality_key = modality[0].lower() if modality not in ("a", "v", "av") else modality

    wer_metric = WER()
    processed = 0
    first_error = None
    csv_dir = os.path.dirname(os.path.abspath(csv_path))

    def resolve_path(raw_path: Optional[str]):
        if raw_path is None:
            return None
        raw_path = raw_path.strip()
        if not raw_path:
            return None
        if os.path.isabs(raw_path):
            return raw_path
        root_candidate = os.path.join(lrs2_root, raw_path)
        if os.path.exists(root_candidate):
            return root_candidate
        return os.path.normpath(os.path.join(csv_dir, raw_path))

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_path", "audio_path", "true_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader, start=1):
            if max_rows > 0 and (row_idx - 1) >= max_rows:
                break

            raw_video = row.get("video_path")
            raw_audio = row.get("audio_path")
            video_path = resolve_path(raw_video if isinstance(raw_video, str) else None)
            audio_path = resolve_path(raw_audio if isinstance(raw_audio, str) else None)
            true_text = (row.get("true_text") or "").strip()

            if not true_text:
                ref_source = video_path if video_path else audio_path
                if not ref_source:
                    raise ValueError(f"row={row_idx} has neither video_path nor audio_path")
                true_text = read_first_line(to_txt_path(ref_source))

            try:
                if modality_key == "a":
                    pred_text = transcribe(
                        audio_path=audio_path, cfg=cfg, modality=modality, device=device, detector=detector
                    )
                elif modality_key == "v":
                    pred_text = transcribe(
                        video_path=video_path, cfg=cfg, modality=modality, device=device, detector=detector
                    )
                else:
                    pred_text = transcribe(
                        video_path=video_path, audio_path=audio_path, cfg=cfg,
                        modality=modality, device=device, detector=detector
                    )

                wer_metric.update(pred_text, true_text)
                processed += 1
            except Exception as exc:
                if first_error is None:
                    first_error = (
                        f"row={row_idx}, video={video_path}, audio={audio_path}, "
                        f"error={type(exc).__name__}: {exc}"
                    )
                continue

    if processed == 0:
        msg = "No valid rows were processed for WER calculation."
        if first_error:
            msg += f" First error: {first_error}"
        raise RuntimeError(msg)

    corpus_wer = wer_metric.compute().item()
    print(f"\n{'='*60}")
    print(" CSV evaluation complete")
    print(f" CSV path   : {csv_path}")
    print(f" Rows mode  : {'top ' + str(max_rows) if max_rows > 0 else 'all rows'}")
    print(f" Rows used  : {processed}")
    print(f" Corpus WER : {corpus_wer:.6f}")

    output_file = f"inference_results_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write("WER EVALUATION RESULTS\n")
        f.write(f"DATE TIME  : {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Corpus WER   : {100*corpus_wer:.4f}%\n")
        f.write(f"{'='*60}\n")

    if first_error:
        print(f" First error: {first_error}")
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------2--

# Register custom resolvers so Hydra config doesn't fail on missing model keys
OmegaConf.register_new_resolver("len", len, replace=True)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Pull --video/--audio and --modality from Hydra overrides (or config)
    video_path = cfg.get("video")
    audio_path = cfg.get("audio")
    checkpoint = cfg.get("model", {}).get("pretrained_model_path")
    modality = cfg.get("modality", "av")
    modality_key = modality[0].lower() if modality not in ("a", "v", "av") else modality
    use_csv_eval = bool(cfg.get("use_csv_eval", False))

    if modality_key not in ("a", "v", "av"):
        print(f"Error: invalid modality '{modality}'. Use one of: av, a, v")
        sys.exit(1)

    if not use_csv_eval and modality_key in ("v", "av") and not video_path:
        print("Error: --video is required for modality=v/av. Usage:")
        print("  python inference.py modality=v video=path/to/video.mp4 model.pretrained_model_path=path/to/model.pth")
        print("  OR")
        print("  python inference.py modality=av video=path/to/video.mp4 audio=path/to/audio.wav model.pretrained_model_path=path/to/model.pth")
        print("  OR")
        print("  python inference.py modality=a audio=path/to/audio.wav model.pretrained_model_path=path/to/model.pth")
        print("  OR")
        print("  python inference.py use_csv_eval=true csv_path=csv/lrs2_datapath.csv")
        sys.exit(1)

    if not use_csv_eval and modality_key == "a" and not audio_path:
        print("Error: --audio is required for modality=a. Usage:")
        print("  python inference.py modality=a audio=path/to/audio.wav model.pretrained_model_path=path/to/model.pth")
        sys.exit(1)

    if not use_csv_eval and modality_key == "av" and not audio_path:
        audio_path = video_path
        log.warning("AV modality: audio_path missing, fallback to video_path for audio decode.")

    if not checkpoint:
        print("Error: model.pretrained_model_path is required.")
        sys.exit(1)

    detector = cfg.get("detector", "mediapipe")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_csv_eval:
        # Keep terminal output minimal during batch CSV runs.
        logging.disable(logging.CRITICAL)
        evaluate_from_csv(cfg, device=device, modality=modality, detector=detector)
        return

    pred_text = transcribe(
        video_path=video_path,
        audio_path=audio_path,
        cfg=cfg,
        modality=modality,
        device=device,
        detector=detector,
    )

    ref_source = audio_path if modality_key == "a" else video_path
    text = read_first_line(to_txt_path(ref_source))
    if text.lower().startswith("text:"):
        text = text.split(":", 1)[1].strip()

    WER_score_sample = WER()
    WER_score_sample.update(pred_text, text)
    score = WER_score_sample.compute().item()

    output_file = f"inference_results_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{ref_source}: Text= {text} / Predicted Text= {pred_text}\n")
        f.write(f"WER: {100 * score:.4f}%\n")

    print(f"\n{'='*60}")
    print(f" Modality : {modality}")
    print(f" Video    : {video_path if video_path else '-'}")
    print(f" Audio    : {audio_path if audio_path else '-'}")
    print(f" Result   : {pred_text}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
