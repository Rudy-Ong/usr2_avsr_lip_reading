# Running `inference_test_data.ipynb`

This document describes how to run the notebook `inference_test_data.ipynb` (resumable batched inference), how to quickly smoke-test it, and where to look for outputs and validation results.

**Files referenced**
- Notebook: `inference_test_data.ipynb`
- Configs: `conf/` (Hydra config dir)
- Scripts: `utils/inference_.py` and `inference.py` (notebook imports `utils.inference_`)
- CSV inputs/outputs: `csv/` (e.g. `csv/lrs2_test_modified.csv` and generated outputs)

**Goal**
Run batched inference reliably (audio-only, video-only or AV) while keeping resume/checkpoint behavior and correct WER computation.

Prerequisites
- Python 3.8+ with the project `requirements.txt` installed.
- (Optional) GPU + CUDA if you want to use `cuda` device.
- Your default shell is `csh`. The examples below are written for `csh` where necessary.

Quick environment setup (recommended)
- Using virtualenv (csh):

```csh
python -m venv .venv
source .venv/bin/activate.csh
pip install -r requirements.txt
```

- Alternatively, use conda/mamba if you prefer.

Set GPU visibility (csh)
```csh
setenv CUDA_VISIBLE_DEVICES 0
```

Launching Jupyter (recommended interactive run)
- From the repository root run (in a tmux session if long-running):

```csh
# start jupyter lab on port 8899
jupyter lab --no-browser --port 8899 &
```

- Open the notebook in your browser and follow the cell order below.

Headless execution (non-interactive)
- To execute the entire notebook end-to-end (useful for CI/smoke runs), use `nbconvert`:

```sh
jupyter nbconvert --to notebook --execute inference_test_data.ipynb \
  --ExecutePreprocessor.timeout=6000 --output executed_inference.ipynb
```

- If you prefer `papermill` (for parameterized runs) you can use `papermill` as well.

Recommended cell execution order (manual, interactive)
The notebook is organized into logical blocks — run these cells in order to avoid missing state or config:

1. Cell 3 — Imports and `inference_module` reload: loads `utils.inference_` and sets `transcribe = inference_module.transcribe`. Run this first so you get the updated `transcribe()` implementation.
2. Cell 4 — (Optional) `wandb.login()` if you want W&B logging; only run if you use W&B.
3. Cell 6 — Hydra config and runtime variables: sets `modality`, `CSV_IN`, `CSV_OUT`, `MAX_ROWS`, etc. Edit `modality` and `MAX_ROWS` here before running the long loop.
   - `modality` can be `"a"` (audio), `"v"` (video), or `"av"` (audio+video).
   - Set `MAX_ROWS` to a small number (e.g. `10`) for a quick smoke test.
4. Cell 9 — Helper utilities: `normalize_text`, `load_checkpoint`, `save_checkpoint`, `resolve_audio_path`.
5. Cell 11 — `read_lrs2_test_csv()` and initial DataFrame preparation. This cell reads the input CSV and prepares the `df` plus checkpoint state.
6. Cell 13 — Validation checks. Confirms `CSV_IN` exists and checks a small sample of audio files.
7. Cell 17 — (Optional) Compose the Hydra config with model overrides (backbone, pretrained checkpoint). Required if you need custom model overrides before running the main loop.
8. Cell 18 — Main batched inference loop. This is the long-running cell that iterates the DataFrame, calls `transcribe(...)`, writes `prediction_words`, `wer`, `status`, and maintains the checkpoint. Run this after confirming the previous cells.
9. Cell 20 — `jiwer`-based aggregate WER example (concatenated refs/hyps)
10. Cell 21 — `metrics.WER` corpus-based aggregate WER computation (corpus-level).
11. Cell 22 — Inspect / re-run specific error rows. Use this to re-infer rows listed in `for idx in (...)` — replace `()` with a list of indices to re-run.

Smoke test
1. Edit cell 6: set `MAX_ROWS = 5` (or a small number), set `modality='a'` (or `av` as needed).
2. Run cells in the order above (3 → 6 → 9 → 11 → 13 → 17 (if needed) → 18).
3. Confirm `CSV_OUT` exists and contains predictions and `status` column. The `CHECKPOINT_PATH` JSON is also written periodically.

Notes & tips
- Always run cell 3 (imports) after you change `utils/inference_.py` to reload the module and pick up code edits.
- The notebook now calls `transcribe` with named args from its main loop; do not call `transcribe` positionally in your own experiments unless you understand the signature.
- AV alignment: `utils.inference_` and `inference.py` contain AV audio/video temporal alignment to pad/trim audio to match video-derived target length (samples_per_frame × frames). This avoids off-by-one temporal mismatches that previously caused tensor-size errors during model fusion.

If you see runtime errors like:
```
Sizes of tensors must match except in dimension 2. Expected size X but got size Y for tensor number N in the list
```
Then:
- Ensure you re-ran cell 3 (imports) to load the latest patched `transcribe` that pads/trims audio.
- Re-run the failed row(s) using Cell 22 by setting explicit `idx` values.

How to re-run specific error rows (Cell 22)
- Edit the line `for idx in ():` to include a list of indices. Example:
```py
for idx in [5, 12, 20]:
    ...
```
- Then run the cell to perform inference only for those rows.

Headless debugging and quick retries
- Use `MAX_ROWS` for quick tests.
- Use the per-row `status` column in `CSV_OUT` to find error rows. Re-run them via Cell 22.

What to check after a run
- `CSV_OUT` (e.g. `csv/<modality>_run_<timestamp>.csv`): check `status`, `prediction_words`, and `wer` columns.
- `CHECKPOINT_PATH` JSON: contains `processed` mapping for resume.
- Console output from Cell 18 prints aggregate WER (via `metrics.WER`) and count of rows used.

Optional improvements (debugging suggestions)
- Add an `audio_alignment` debug column when saving rows to show whether audio was 'ok'|'padded'|'trimmed'. Example snippet to set in notebook after receiving `alignment_status` from `transcribe` (if implemented):
```py
# set a debug column value per row
df.at[idx, 'audio_alignment'] = alignment_status  # 'ok'|'padded'|'trimmed'
```
- If you want a complete machine-run, consider adding a small driver script that imports `utils.inference_` and drives the CSV loop; this is easier to run under `tmux` or cron.

Troubleshooting
- If `wandb` errors occur, skip Cell 4 or set `HAS_WANDB=False` in Cell 3.
- If Hydra composition fails, ensure you used the same `config_dir` as the notebook (Cell 6) and that `conf/config.yaml` exists.

Contact / next steps
- After a successful smoke test (small `MAX_ROWS`), run the full CSV by setting `MAX_ROWS = -1` and monitor the tmux/jupyter session. If any rows still error, run Cell 22 with those indices and paste the exception for further debugging.

---
Generated on: April 17, 2026
Disclaimer: This document is auto-generated based on the current state of the repository and may not reflect future changes.
