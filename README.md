# SpeechSR

## Repository structure

```text
SpeechSR/
├── data/
│   ├── coil_sensitivities/
│   ├── kspace/
│   │   ├── full/
│   │   └── undersampled/
│   └── images/
│       ├── HR/
│       ├── Synth_LR/
│       └── Dynamic/
├── models/
│   ├── pretrained/
│   │   ├── README.md
│   │   ├── unet.pth
│   │   ├── proposed.pth
│   │   ├── srcnn.pth
│   │   ├── vdsr.pth
│   │   └── edsr.pth
│   ├── edsr.py
│   ├── losses.py
│   ├── srcnn.py
│   ├── proposed.py
│   ├── vdsr.py
│   └── unet.py
├── utils/
│   ├── evaluation_boxplots.py
│   ├── intensity_time_plotter.py
│   ├── nifti_to_png.py
│   ├── sense_reconstruction.py
│   └── synthetic_undersampling.py
├── infer.py
├── train.py
├── colab_training.ipynb
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data conventions

- Input folder: `data/images/Synth_LR`
- Target folder: `data/images/HR`
- Filenames should contain subject IDs in the form `SubjectXXXX`.
- `train.py` pairs LR/HR files by name with safe fallbacks (`LR_ -> HR_`, `Synth_LR -> HR`, etc.).

## Training

### 1) Train U-Net

```bash
python train.py \
  --model unet \
  --input-dir data/images/Synth_LR \
  --target-dir data/images/HR \
  --output-dir outputs/unet \
  --epochs 200
```

### 2) Train Proposed

```bash
python train.py \
  --model proposed \
  --input-dir data/images/Synth_LR \
  --target-dir data/images/HR \
  --output-dir outputs/proposed \
  --proposed-target-size 1024 \
  --epochs 200
```

### 3) Hyperparameter optimisation with Optuna

```bash
python train.py \
  --model proposed \
  --input-dir data/images/Synth_LR \
  --target-dir data/images/HR \
  --output-dir outputs/proposed_hpo \
  --use-optuna \
  --n-trials 30 \
  --hpo-epochs 25 \
  --train-after-hpo
```

`best_hyperparameters.json` and `optuna_study.db` are saved in the chosen output directory.

## Colab training (single-script workflow)

You can either run commands manually in Colab or open the ready notebook:
- `colab_training.ipynb`

1. Upload this repo to GitHub (`marioknicola/SpeechSR`) and open Colab.
2. In Colab, clone and install:

```bash
git clone https://github.com/marioknicola/SpeechSR.git
cd SpeechSR
pip install -r requirements.txt
```

3. Mount Drive and point `--input-dir` / `--target-dir` to your private folders:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
python train.py \
  --model unet \
  --input-dir /content/drive/MyDrive/your_data/Synth_LR \
  --target-dir /content/drive/MyDrive/your_data/HR \
  --output-dir /content/drive/MyDrive/SpeechSR_outputs/unet \
  --epochs 200
```

Ensure that a GPU runtime is enabled (top-left of the screen). The T4 is allowed for free.

## Utilities

- `utils/synthetic_undersampling.py`: Generate truncated k-space `.mat` from full k-space.
- `utils/sense_reconstruction.py`: Run generalized SENSE reconstruction from k-space + coil sensitivity maps.
- `utils/nifti_to_png.py`: Convert NIfTI slices/frames to PNG by duplicating across channels.
- `utils/intensity_time_plotter.py`: Generate M-mode intensity-time plots. There is functionality to use the same profiles and temporal CNR ROIs for multiple files
- `utils/evaluation_boxplots.py`: Generate PSNR/SSIM/MSE boxplots and CSV file.

## Pretrained models

Tracked checkpoints currently in `models/pretrained/`:
- `unet.pth`
- `proposed.pth`
- `srcnn.pth`
- `vdsr.pth`
- `edsr.pth`

## Inference

Run inference with any pretrained model using `infer.py`:

```bash
python infer.py \
  --model unet \
  --input data/images/Synth_LR/LR_kspace_Subject0021_oh.nii \
  --output outputs/inference/unet_Subject0021_oh.nii
```

Supported models are:
- `unet`
- `proposed`
- `srcnn`
- `vdsr`
- `edsr`

For model-specific command examples and checkpoint conventions, see:
- `models/pretrained/README.md`

## Inference result figures (add your own)

Paste your images below after pushing or locally previewing in Markdown:

```markdown
## Sample Inference Results

![U-Net qualitative result](path/to/unet_result.png)
![Proposed qualitative result](path/to/proposed_result.png)
![Comparison panel](path/to/comparison_panel.png)
```

## Notes

- `.nii`, `.nii.gz`, and `.mat` are ignored for repository size control.
- `.pth` files in `models/pretrained/` are tracked by design.
- If your previous naming differs, adapt the pairing logic in `PairedMRIDataset._find_target` inside `train.py`.
