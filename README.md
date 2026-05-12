# AI Sign Language Recognition (ai-sign_trans)

This repository contains the AI code for Arabic sign language recognition (code-only). It intentionally excludes datasets and trained model weights.

Highlights
- Keypoint extraction using MediaPipe
- Multiple model variants: CNN+BiLSTM, Multi-stream, Hybrid Graph-Transformer
- Few-shot training and evaluation pipelines

Quick Start
1. Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or: .\.venv\Scripts\activate   # cmd.exe
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Typical workflow (code-only — provide your dataset paths locally):

- Extract keypoints from videos:

```bash
python extract_keypoints.py --input_dir /path/to/videos --output_dir keypoints_output
```

- Preprocess keypoints:

```bash
python preprocess.py --input_dir keypoints_output --output_dir keypoints_normalized
```

- Train a baseline model:

```bash
python train.py --manifest splits/train_manifest.csv --val_manifest splits/val_manifest.csv
```

- Evaluate a model:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --manifest splits/test_manifest.csv
```

- Run inference (example):

```bash
python inference.py --model model_export/arsl_cnn_bilstm.pt --input sample_keypoints.npy
```

Notes
- Do NOT commit datasets, keypoints, or checkpoints. Those paths are in .gitignore.
- If your dataset is large, create a sanitized sample manifest before publishing.

Contributing
- Open an issue or create a PR. Keep data and private artifacts out of the repo.

License & Contact
- Add your LICENSE file if you plan to publish publicly.
- For help, contact the project owner.

