"""
Pretrain the SpeechSR generator on fastMRI data.

fastMRI provides raw multi-coil k-space. This script:
  1. Loads .h5 files from a fastMRI download directory.
  2. Reconstructs each slice as a magnitude image (root sum-of-squares) to get
     the HR target, then k-space zero-pads it to 1024×1024 — matching exactly
     the training pipeline in train.py.
  3. Simulates undersampling by k-space truncation to produce 128×128 LR input
     (centre-crop of k-space then IFFT), mirroring the speech MRI pipeline.
  4. Trains ProposedModel on LR → HR pairs using ForegroundEdgePerceptualLoss.
  5. Saves a checkpoint for subsequent fine-tuning with train.py or train_gan.py.

Download fastMRI at: https://fastmri.org/dataset/
  Recommended starting point: brain_singlecoil_train (simpler, no coil combining needed)
  or brain_multicoil_train (richer, but requires coil combining — both are handled here).

Requirements (install separately):
    pip install h5py

Usage:
    python pretrain_fastmri.py \\
        --data-dir /path/to/fastmri/brain_singlecoil_train \\
        --output-dir outputs/fastmri_pretrained \\
        --epochs 50

    # Then fine-tune on your speech MRI subjects:
    python train.py --model proposed \\
        --input-dir data/images/Synth_LR \\
        --target-dir data/images/HR \\
        --output-dir outputs/finetuned \\
        --generator-checkpoint outputs/fastmri_pretrained/best_model.pth \\
        --patch-size 64 --augment --perceptual-loss \\
        --epochs 200
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from models import ForegroundEdgePerceptualLoss, build_proposed
from train import normalize01, pick_device, set_seed

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# k-space utilities
# ---------------------------------------------------------------------------

def kspace_truncate(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Simulate k-space undersampling by centre-cropping the 2D FFT.
    This is the inverse of kspace_zeropad in train.py: the centre
    target_size × target_size lines of k-space are kept and the rest
    discarded, then IFFTed back to image space (Sinc decimation).
    """
    h, w = image.shape
    tensor = torch.from_numpy(image.astype(np.float32))
    kspace = torch.fft.fftshift(torch.fft.fft2(tensor))
    ch, cw = h // 2, w // 2
    half = target_size // 2
    truncated = kspace[ch - half:ch + half, cw - half:cw + half]
    output = torch.fft.ifft2(torch.fft.ifftshift(truncated)).abs().cpu().numpy()
    return normalize01(output)


def kspace_zeropad(image: np.ndarray, target_size: int) -> np.ndarray:
    """Zero-pad k-space to target_size (Sinc interpolation upsampling)."""
    h, w = image.shape
    tensor = torch.from_numpy(image.astype(np.float32))
    kspace = torch.fft.fftshift(torch.fft.fft2(tensor))
    padded = torch.zeros((target_size, target_size), dtype=torch.complex64)
    rs = (target_size - h) // 2
    cs = (target_size - w) // 2
    padded[rs:rs + h, cs:cs + w] = kspace
    output = torch.fft.ifft2(torch.fft.ifftshift(padded)).abs().cpu().numpy()
    return normalize01(output)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FastMRIDataset(Dataset):
    """
    Loads slices from fastMRI HDF5 files and produces LR/HR pairs.

    Each HDF5 file contains a 3D volume. We extract every slice, reconstruct
    the HR image from k-space (RSS for multi-coil, magnitude for single-coil),
    then synthesise a 128×128 LR input via k-space truncation.

    The resulting pairs match the SpeechSR training format:
        LR: 128×128 magnitude image (tensor shape 3×128×128 in temporal mode)
        HR: 512×512 k-space zero-padded image (tensor shape 1×512×512)
    """

    def __init__(
        self,
        data_dir: Path,
        target_size: int = 512,
        lr_size: int = 128,
        patch_size: int | None = None,
        augment: bool = False,
        max_files: int | None = None,
        in_channels: int = 3,
    ) -> None:
        if not HAS_H5PY:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.target_size = target_size
        self.lr_size = lr_size
        self.patch_size = patch_size
        self.augment = augment
        self.in_channels = in_channels

        h5_files = sorted(data_dir.glob("*.h5"))
        if max_files is not None:
            h5_files = h5_files[:max_files]
        if not h5_files:
            raise ValueError(f"No .h5 files found in {data_dir}")

        # Build flat list of (file, slice_idx) pairs
        self.samples: list[tuple[Path, int]] = []
        for f in h5_files:
            try:
                with h5py.File(f, "r") as hf:
                    n_slices = self._n_slices(hf)
                    self.samples.extend((f, i) for i in range(n_slices))
            except Exception:
                continue  # skip corrupt files

        if not self.samples:
            raise ValueError("No valid slices found in dataset.")

    def _n_slices(self, hf: "h5py.File") -> int:
        if "kspace" in hf:
            return hf["kspace"].shape[0]
        if "reconstruction_rss" in hf:
            return hf["reconstruction_rss"].shape[0]
        return 0

    def _load_slice(self, path: Path, idx: int) -> np.ndarray:
        """Return a 2D float32 magnitude image for the given slice."""
        with h5py.File(path, "r") as hf:
            if "reconstruction_rss" in hf:
                # Pre-computed RSS reconstruction — use directly
                img = hf["reconstruction_rss"][idx]  # (H, W) float32
                return normalize01(np.array(img, dtype=np.float32))

            # Raw k-space: reconstruct RSS from all coils
            kspace = np.array(hf["kspace"][idx])  # (coils, H, W) complex64 or (H, W)
            if kspace.ndim == 2:
                kspace = kspace[np.newaxis]  # treat as single coil
            # IFFT each coil image, then RSS combine
            coil_images = np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)))
            rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)).astype(np.float32)
            return normalize01(rss)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, slice_idx = self.samples[idx]
        hr_full = self._load_slice(path, slice_idx)   # (H, W) normalised float32

        # Resize to a canonical 512×512 before applying k-space ops
        # (fastMRI brain slices vary in size; 512 matches the speech MRI HR size)
        if hr_full.shape != (512, 512):
            t = torch.from_numpy(hr_full).unsqueeze(0).unsqueeze(0)
            hr_full = F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False).squeeze().numpy()

        lr = kspace_truncate(hr_full, self.lr_size)       # 128×128
        hr = kspace_zeropad(hr_full, self.target_size)    # 1024×1024

        # For temporal mode, repeat single frame across all 3 channels so the
        # head weights are primed; real temporal context comes from dynamic data.
        if self.in_channels == 3:
            lr_t = torch.from_numpy(lr).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        else:
            lr_t = torch.from_numpy(lr).unsqueeze(0)                  # (1, H, W)
        hr_t = torch.from_numpy(hr).unsqueeze(0)   # (1, target_size, target_size)

        if self.patch_size is not None:
            lr_t, hr_t = self._random_patch(lr_t, hr_t)

        if self.augment and random.random() < 0.5:
            lr_t = torch.flip(lr_t, dims=[2])
            hr_t = torch.flip(hr_t, dims=[2])

        return {"input": lr_t, "target": hr_t}

    def _random_patch(self, lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, lh, lw = lr.shape
        _, hh, _ = hr.shape
        scale = hh // lh
        ps = self.patch_size
        if lh < ps or lw < ps:
            return lr, hr
        y0 = random.randint(0, lh - ps)
        x0 = random.randint(0, lw - ps)
        return (
            lr[:, y0:y0 + ps, x0:x0 + ps],
            hr[:, y0 * scale:(y0 + ps) * scale, x0 * scale:(x0 + ps) * scale],
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            loss = criterion(model(inputs), targets)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
        running_loss += float(loss.item())
    return running_loss / max(1, len(dataloader))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*.h5"))
    if not all_files:
        raise ValueError(f"No .h5 files found in {data_dir}. Check --data-dir.")

    # Apply --max-files limit before the train/val split
    max_files = getattr(args, "max_files", None)
    if max_files is not None:
        all_files = all_files[:max_files]
        print(f"Using {len(all_files)} .h5 files (--max-files={max_files}).", flush=True)

    # Hold out the last 10% of files for validation
    n_val = max(1, len(all_files) // 10)
    train_files = all_files[:-n_val]
    val_files = all_files[-n_val:]

    patch_size = args.patch_size if args.patch_size > 0 else None

    in_channels = getattr(args, "in_channels", 3)
    train_dataset = FastMRIDataset(
        data_dir=data_dir,
        target_size=args.target_size,
        lr_size=args.lr_size,
        patch_size=patch_size,
        augment=args.augment,
        max_files=len(train_files),
        in_channels=in_channels,
    )
    val_dataset = FastMRIDataset(
        data_dir=data_dir,
        target_size=args.target_size,
        lr_size=args.lr_size,
        max_files=len(val_files),
        in_channels=in_channels,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    in_channels = getattr(args, "in_channels", 3)
    model = build_proposed(n_res_blocks=16, n_feats=64, reduction=16, res_scale=0.1, in_channels=in_channels).to(device)

    # Optionally warm-start from an existing checkpoint
    if args.generator_checkpoint:
        ckpt_path = Path(args.generator_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Generator checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded generator from {ckpt_path}")

    criterion = ForegroundEdgePerceptualLoss(alpha_percep=args.alpha_percep)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    best_path = output_dir / "best_model.pth"
    history = []

    print(f"Training on {len(train_dataset)} slices, validating on {len(val_dataset)} slices.")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_l = run_epoch(model, train_loader, criterion, optimizer, device, args.max_grad_norm)
        val_l = run_epoch(model, val_loader, criterion, optimizer=None, device=device, max_grad_norm=0.0)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | train={train_l:.6f} | val={val_l:.6f} | lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_l, "val_loss": val_l})
        if val_l < best_val:
            best_val = val_l
            torch.save({"model_state_dict": model.state_dict()}, best_path)

    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"Pretraining complete. Best checkpoint: {best_path}  (val_loss={best_val:.6f})")
    print("Fine-tune on speech MRI subjects using train.py --generator-checkpoint.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain SpeechSR on fastMRI data")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing fastMRI .h5 files")
    parser.add_argument("--output-dir", type=str, default="outputs/fastmri_pretrained")
    parser.add_argument("--generator-checkpoint", type=str, default=None,
                        help="Optional existing checkpoint to warm-start from.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-size", type=int, default=512,
                        help="HR output size in pixels (default 512 matches speech MRI pipeline).")
    parser.add_argument("--lr-size", type=int, default=128, help="LR input size (k-space truncated).")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of .h5 files used (e.g. 150 for a manageable subset). "
                             "Files are selected from the start of the sorted list.")
    parser.add_argument("--in-channels", type=int, default=3, choices=[1, 3],
                        help="1 = single-frame, 3 = temporal (default, matches train_gan.py).")
    parser.add_argument("--patch-size", type=int, default=64,
                        help="LR patch size for patch-based training. 0 to use full images.")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--alpha-percep", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
