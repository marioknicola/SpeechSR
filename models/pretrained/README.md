# Pretrained Models

This folder stores the tracked pretrained checkpoints used by `SpeechSR/infer.py`.

Current checkpoint names:
- `unet.pth`
- `proposed.pth`
- `srcnn.pth`
- `vdsr.pth`
- `edsr.pth`

## Colab training notebook

Use [colab_training.ipynb](../../colab_training.ipynb) from the repository root to train `unet` or `proposed` in Google Colab.

## Inference (all models)

From the repository root:

```bash
python infer.py --help
```

Required args:
- `--model`: `unet`, `proposed`, `srcnn`, `vdsr`, `edsr`
- `--input`: input `.nii`/`.nii.gz`
- `--output`: output `.nii`/`.nii.gz`

If `--checkpoint` is omitted, the default is `models/pretrained/<model>.pth`.

### U-Net

```bash
python infer.py \
	--model unet \
	--input data/images/Synth_LR/LR_kspace_Subject0021_oh.nii \
	--output outputs/inference/unet_Subject0021_oh.nii
```

### Proposed

```bash
python infer.py \
	--model proposed \
	--input data/images/Synth_LR/LR_kspace_Subject0021_oh.nii \
	--output outputs/inference/proposed_Subject0021_oh.nii
```

### SRCNN

```bash
python infer.py \
	--model srcnn \
	--input data/images/Synth_LR/LR_kspace_Subject0021_oh.nii \
	--output outputs/inference/srcnn_Subject0021_oh.nii
```

### VDSR

```bash
python infer.py \
	--model vdsr \
	--input data/images/Synth_LR/LR_kspace_Subject0021_oh.nii \
	--output outputs/inference/vdsr_Subject0021_oh.nii
```

### EDSR

```bash
python infer.py \
	--model edsr \
	--input data/images/Synth_LR/LR_kspace_Subject0021_oh.nii \
	--output outputs/inference/edsr_Subject0021_oh.nii
```

## Optional explicit checkpoint

```bash
python infer.py \
	--model proposed \
	--checkpoint models/pretrained/proposed.pth \
	--input data/images/Synth_LR/LR_kspace_Subject0021_oh.nii \
	--output outputs/inference/proposed_Subject0021_oh.nii
```