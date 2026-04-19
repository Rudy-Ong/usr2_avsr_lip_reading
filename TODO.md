# Running `inference_test_data.ipynb`

This document describes how to run the notebook `inference_test_data.ipynb`, how to quickly smoke-test it, and where to look for outputs and validation results.

**Files referenced**
- Notebook: `inference_test_data.ipynb`
- Configs: `conf/` (Hydra config dir)
- Scripts: `utils/inference_.py`
- CSV inputs/outputs: `csv/` (e.g. `csv/your_filepath_here.csv` and generated outputs)

**Checks**
- Ensure you have data files store in the respective folders and write each file path in csv file.
- If feeding video files in the first time, the module mediapipe or retinaface will try to detect face and crop mouth for lip region in main directory. 
- If the face is not detected, the module will raise error and skip that video file.

**Tasks**
Run batched inference reliably (audio-only, video-only or AV) while keeping resume/checkpoint behavior and correct WER computation.

Quick environment setup (recommended)
- Using virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate.csh
pip install -r requirements.txt
```

Open Jupyter Notebook and run `inference_test_data.ipynb` cells in order 

Cell execution order :
Run this notebook cells in order to avoid missing state or config, following are brief explanation of each cell contents:
1. Cell 2 — Imports and `inference_module` reload: loads `utils.inference_` and sets `transcribe = inference_module.transcribe`. Run this first so you get the updated `transcribe()` implementation.
2. Cell 3 — (Optional) `wandb.login()` if you want W&B logging; only run if you use W&B.
3. Cell 5 — Hydra config and runtime variables: sets `modality`, `CSV_IN`, `CSV_OUT`, `MAX_ROWS`, etc. Edit `modality` and `MAX_ROWS` here before running the long loop.
   - `modality` can be `"a"` (audio), `"v"` (video), or `"av"` (audio+video).
   - Set `MAX_ROWS` to a small number (e.g. `10`) for a quick smoke test.
4. Cell 7 — Helper utilities: `normalize_text`, `load_checkpoint`, `save_checkpoint`, etc.
5. Cell 9 — `read_test_csv()` and initial DataFrame preparation.
6. Cell 11 — Paths validation checks. Confirms `CSV_IN` exists and checks a small sample of input files.
7. Cell 15 — (Optional) Compose the Hydra config with model overrides (resnet backbone, pretrained checkpoint). Required if you need custom model overrides before running the main loop.
8. Cell 16 — Main batched inference loop. This cell calls `transcribe(...)`, writes `prediction_words`, `wer`, `status`, and maintains the checkpoint. Outputs are saved to `CSV_OUT` path in form of CSV and JSON.
9.  Cell 17 — `metrics.WER` corpus-based aggregate WER computation (corpus-level).