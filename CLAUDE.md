# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

USR 2.0 is a multimodal audio-visual speech recognition system (inference-only). Built on PyTorch Lightning + Hydra + Weights & Biases.

## Key Commands

### Environment Setup
```bash
conda env create -f environment.yml
```

### Evaluation
```bash
python main.py \
  model/backbone=resnet_transformer_base \
  model.pretrained_model_path=<checkpoint.pth> \
  data.dataset.test_csv=<test.csv> \
  decode.beam_size=1 decode.ctc_weight=0.0
```

### Hydra Config Overrides
All configuration is in `conf/` with `conf/config.yaml` as root. Override any value on the command line:
```bash
python main.py model/backbone=resnet_transformer_large
```

## Architecture

### Entry Point
- `main.py` — runs test/inference

### Core Modules
- `evaluator.py` — PyTorch Lightning module (`USREvaluator`) for inference (beam search decoding, WER computation)
- `models/usr.py` — `USRModel` and `USR` wrapper: multimodal encoder with CTC head

### Data Pipeline
- `data/datamodule.py` — `USRDataModule` Lightning DataModule (test dataloader only)
- `data/dataset.py` — `AVDataset` loading from CSVs pointing to video/audio files
- `data/transforms.py` — audio/video transforms (`NormalizeVideo`, `AddNoise`)
- `data/samplers.py` — frame-count-based sampling for variable-length sequences

### Model Backbone (espnet/)
Vendored ESPnet components:
- `espnet/nets/pytorch_backend/backbones/` — visual frontends (ResNet, ShuffleNetV2, Conv1D/3D)
- `espnet/nets/pytorch_backend/transformer/` — transformer encoder/decoder
- `espnet/nets/pytorch_backend/e2e_asr_transformer.py` — end-to-end ASR transformer
- `espnet/nets/pytorch_backend/ctc.py` — CTC loss
- `espnet/nets/batch_beam_search.py` — beam search decoding

### Config Structure (conf/)
Hydra config groups: `data/`, `decode/`, `model/`, `logger/`, `logging/`.

Dataset paths must be set in `conf/data/default.yaml`.

### Backbone Options
Set via `model/backbone=`: `resnet_transformer_base`, `resnet_transformer_baseplus`, `resnet_transformer_large`, `resnet_transformer_huge`.

### Utilities
- `utils/utils.py` — tokenization (`UNIGRAM1000_LIST`), `ids_to_str`, `set_requires_grad`
