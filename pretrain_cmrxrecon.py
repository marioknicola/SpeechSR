"""
Pretrain the SpeechSR generator on CMRxRecon cardiac cine MRI data.

CMRxRecon (MICCAI 2023/2024 challenge) provides fully-sampled multi-coil k-space
for cardiac cine MRI (short-axis and long-axis views). Each acquisition is a
temporal sequence of cardiac phases (~10–30 frames per heartbeat cycle).

Why CMRxRecon is better than fastMRI for this task:
  1. Temporal: cardiac cine frames give *real* temporal triplets (t-1, t, t+1)
     so the temporal consistency loss trains on genuine motion, not repeated statics.
  2. Dynamic: acquisition modality is closest to speech MRI — both are dynamic,
     undersampled, and reconstructed from multi-coil k-space.
  3. K-space structure: the same truncation / zero-padding pipeline applies
     with no domain-format gap.

The script:
  1. Walks a CMRxRecon directory for .mat files (v5 or v7.3/HDF5).
  2. Reconstructs each (slice, frame) as a magnitude image via root sum-of-squares.
  3. Synthesises 128×128 LR via k-space truncation and 512×512 HR via zero-padding.
  4. Each training step draws:
       (a) a supervised (LR_t, HR_t) pair — ForegroundEdgePerceptualLoss
       (b) a temporal window (LR_t-1, LR_t, LR_t+1) — TemporalConsistencyLoss
           + KSpaceConsistencyLoss on the SR output
  5. Saves a checkpoint for warm-starting train_gan.py.

Download:
  Register at https://www.synapse.org  and obtain a personal access token.
  See the Colab notebook (Section 2) for the download cell.

  Dataset page:  https://cmrxrecon.github.io/
  Synapse ID:    syn51471091  (CMRxRecon 2023, MultiCoil Cine)

Requirements (additional to requirements.txt):
    pip install synapseclient   # for Colab download only
    (h5py already in requirements.txt)

Usage:
    python pretrain_cmrxrecon.py \\
        --data-dir /path/to/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample \\
        --output-dir outputs/cmrxrecon_pretrained \\
        --epochs 30 \\
        --max-files 100

    # Then fine-tune on speech MRI:
    python train_gan.py \\
        --generator-checkpoint outputs/cmrxrecon_pretrained/best_model.pth \\
        --temporal --dynamic-dir data/images/Dynamic_128 \\
        --pretrain-epochs 150 --gan-epochs 0 \\
        --input-dir data/images/Synth_LR --target-dir data/images/HR
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from models import (
    ForegroundEdgePerceptualLoss,
    KSpaceConsistencyLoss,
    TemporalConsistencyLoss,
    build_proposed,
)
from train import normalize01, pick_device, set_seed

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# k-space utilities (identical to train.py / pretrain_fastmri.py)
# ---------------------------------------------------------------------------

def kspace_truncate(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape
    tensor = torch.from_numpy(image.astype(np.float32))
    kspace = torch.fft.fftshift(torch.fft.fft2(tensor))
    ch, cw = h // 2, w // 2
    half = target_size // 2
    truncated = kspace[ch - half:ch + half, cw - half:cw + half]
    return normalize01(torch.fft.ifft2(torch.fft.ifftshift(truncated)).abs().cpu().numpy())


def kspace_zeropad(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape
    tensor = torch.from_numpy(image.astype(np.float32))
    kspace = torch.fft.fftshift(torch.fft.fft2(tensor))
    padded = torch.zeros((target_size, target_size), dtype=torch.complex64)
    rs = (target_size - h) // 2
    cs = (target_size - w) // 2
    padded[rs:rs + h, cs:cs + w] = kspace
    return normalize01(torch.fft.ifft2(torch.fft.ifftshift(padded)).abs().cpu().numpy())


# ---------------------------------------------------------------------------
# CMRxRecon .mat file loader
# ---------------------------------------------------------------------------

def _load_mat_kspace(path: Path) -> np.ndarray | None:
    """
    Load k-space from a CMRxRecon .mat file.

    CMRxRecon .mat files contain a variable named 'kspace_full' (fully-sampled
    multi-coil k-space). Files may be MATLAB v5 (scipy) or v7.3/HDF5 (h5py).

    Returns a complex128 array with shape inferred as (coils, frames, slices, ky, kx).
    If the file cannot be parsed, returns None.
    """
    # ------------------------------------------------------------------
    # Try HDF5 / MATLAB v7.3 first (most CMRxRecon files are this format)
    # ------------------------------------------------------------------
    if HAS_H5PY:
        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                kspace_key = next((k for k in keys if "kspace" in k.lower()), None)
                if kspace_key is None:
                    return None
                raw = f[kspace_key]
                # MATLAB stores complex as a structured dtype with 'r' and 'i' fields,
                # or directly as complex64/complex128 depending on MATLAB version.
                if raw.dtype.names and "r" in raw.dtype.names:
                    kspace = raw["r"][()] + 1j * raw["i"][()]
                elif np.issubdtype(raw.dtype, np.complexfloating):
                    kspace = raw[()]
                else:
                    # Some exports store real/imag as separate datasets
                    kspace = raw[()].astype(np.complex64)
                return kspace.astype(np.complex64)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Fall back to scipy (MATLAB v5)
    # ------------------------------------------------------------------
    if HAS_SCIPY:
        try:
            mat = scipy.io.loadmat(str(path))
            kspace_key = next((k for k in mat if "kspace" in k.lower() and not k.startswith("_")), None)
            if kspace_key is None:
                return None
            return mat[kspace_key].astype(np.complex64)
        except Exception:
            pass

    return None


def _rss_reconstruct(kspace_2d_multi: np.ndarray) -> np.ndarray:
    """
    Root sum-of-squares reconstruction from a (coils, ky, kx) complex k-space.
    Returns a 2D float magnitude image.
    """
    coil_images = np.fft.ifft2(np.fft.ifftshift(kspace_2d_multi, axes=(-2, -1)))
    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)).astype(np.float32)
    return normalize01(rss)


def _parse_kspace_shape(kspace: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """
    Infer (n_coils, n_frames, n_slices, ky, kx) from an array of unknown axis order.

    CMRxRecon 2023 stores k-space as (nt, nz, nc, ny, nx) but axis order can vary
    across versions and export scripts. We infer by assuming:
      - The two largest dimensions are ky/kx (spatial).
      - The smallest dimension is likely slices (nz ≤ 20 typical).
      - The next smallest is likely frames (nt ≤ 50 typical).
      - The remaining dimension is coils.
    This is a heuristic — override with --kspace-axes if needed.
    """
    if kspace.ndim == 4:
        # (nc, nt_or_nz, ky, kx) — treat as single-slice with coils
        nc, nt, ky, kx = kspace.shape
        return nc, nt, 1, ky, kx
    if kspace.ndim == 5:
        dims = list(kspace.shape)
        spatial_idx = sorted(range(5), key=lambda i: dims[i], reverse=True)[:2]
        ky_ax, kx_ax = sorted(spatial_idx)
        remaining = [i for i in range(5) if i not in (ky_ax, kx_ax)]
        remaining_dims = [(dims[i], i) for i in remaining]
        remaining_dims.sort()  # ascending size
        nz_ax = remaining_dims[0][1]
        nt_ax = remaining_dims[1][1]
        nc_ax = remaining_dims[2][1]
        # Reorder to (nc, nt, nz, ky, kx)
        kspace_reorder = np.moveaxis(kspace, [nc_ax, nt_ax, nz_ax, ky_ax, kx_ax], [0, 1, 2, 3, 4])
        nc, nt, nz, ky, kx = kspace_reorder.shape
        return nc, nt, nz, ky, kx
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CMRxReconDataset(Dataset):
    """
    Cardiac cine MRI pretraining dataset from CMRxRecon.

    Each sample yields both a supervised pair AND a temporal triplet from
    the same cine sequence:

        supervised:  {"lr": (1, 128, 128), "hr": (1, 512, 512)}
        temporal:    {"window_t":  (3, 128, 128),   # (t-1, t, t+1)
                      "window_t1": (3, 128, 128)}   # (t, t+1, t+2)

    The temporal triplets use real cardiac motion — significantly better
    for temporal consistency pretraining than repeated static frames.
    """

    # Canonical size before applying k-space ops
    _CANONICAL_HR = 256

    def __init__(
        self,
        data_dir: Path,
        target_size: int = 512,
        lr_size: int = 128,
        augment: bool = False,
        max_files: int | None = None,
        in_channels: int = 3,
    ) -> None:
        if not (HAS_H5PY or HAS_SCIPY):
            raise ImportError("h5py or scipy is required. Install with: pip install h5py scipy")

        self.target_size = target_size
        self.lr_size = lr_size
        self.augment = augment
        self.in_channels = in_channels

        mat_files = sorted(data_dir.rglob("*.mat"))
        if max_files is not None:
            mat_files = mat_files[:max_files]
        if not mat_files:
            raise ValueError(f"No .mat files found under {data_dir}")

        # Build sample list: each entry is (mag_image_2d, time_index, all_frames_for_slice)
        # We cache the reconstructed magnitude arrays to avoid re-loading .mat per sample.
        # For large datasets store only (file, slice_idx, frame_idx) and load lazily.
        self.samples: list[tuple[Path, int, int]] = []  # (file, slice_idx, frame_idx)
        self._shape_cache: dict[Path, tuple] = {}

        print(f"Indexing {len(mat_files)} CMRxRecon .mat files...", flush=True)
        for mat_path in mat_files:
            kspace = _load_mat_kspace(mat_path)
            if kspace is None:
                continue
            parsed = _parse_kspace_shape(kspace)
            if parsed is None:
                continue
            nc, nt, nz, ky, kx = parsed
            if nt < 3:
                continue  # need at least 3 frames for temporal triplets
            # Only include frames i ∈ [1, nt-3] so both (i-1,i,i+1) and (i,i+1,i+2) are valid
            for z in range(nz):
                for t in range(1, nt - 2):
                    self.samples.append((mat_path, z, t))
            self._shape_cache[mat_path] = (nc, nt, nz, ky, kx)

        if not self.samples:
            raise ValueError(
                "No valid samples found. Check that .mat files contain 'kspace_full' "
                "with ≥ 3 temporal frames."
            )
        print(f"CMRxReconDataset: {len(self.samples)} samples from {len(mat_files)} files.", flush=True)

    def _load_slice_frame(self, path: Path, slice_idx: int, frame_idx: int) -> np.ndarray:
        """Load and reconstruct one (slice, frame) magnitude image."""
        kspace = _load_mat_kspace(path)
        nc, nt, nz, ky, kx = self._shape_cache[path]
        if kspace.ndim == 5:
            dims = list(kspace.shape)
            # Re-apply the same reordering logic used during indexing
            spatial_idx = sorted(range(5), key=lambda i: dims[i], reverse=True)[:2]
            ky_ax, kx_ax = sorted(spatial_idx)
            remaining = [i for i in range(5) if i not in (ky_ax, kx_ax)]
            remaining_dims = [(dims[i], i) for i in remaining]
            remaining_dims.sort()
            nz_ax = remaining_dims[0][1]
            nt_ax = remaining_dims[1][1]
            nc_ax = remaining_dims[2][1]
            kspace = np.moveaxis(kspace, [nc_ax, nt_ax, nz_ax, ky_ax, kx_ax], [0, 1, 2, 3, 4])
        elif kspace.ndim == 4:
            # (nc, nt, ky, kx) — single slice
            kspace = kspace[:, :, np.newaxis, :, :]

        slice_idx = min(slice_idx, kspace.shape[2] - 1)
        frame_idx = min(frame_idx, kspace.shape[1] - 1)
        kspace_2d = kspace[:, frame_idx, slice_idx, :, :]  # (nc, ky, kx)

        mag = _rss_reconstruct(kspace_2d)
        # Resize to canonical size before k-space operations
        if mag.shape != (self._CANONICAL_HR, self._CANONICAL_HR):
            t = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0)
            mag = F.interpolate(t, size=(self._CANONICAL_HR, self._CANONICAL_HR),
                                mode="bilinear", align_corners=False).squeeze().numpy()
        return normalize01(mag)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, z, t = self.samples[idx]
        nc, nt, nz, ky, kx = self._shape_cache[path]

        # Load the 4 frames needed: (t-1, t, t+1, t+2)
        frames = {
            fi: self._load_slice_frame(path, z, fi)
            for fi in [t - 1, t, t + 1, t + 2]
        }

        hr_mag = frames[t]
        lr = kspace_truncate(hr_mag, self.lr_size)    # (128, 128)
        hr = kspace_zeropad(hr_mag, self.target_size)  # (512, 512)

        # Build temporal windows (for TemporalConsistencyLoss)
        def _lr_frame(fi: int) -> np.ndarray:
            return kspace_truncate(frames[fi], self.lr_size)

        if self.in_channels == 3:
            lr_t = torch.from_numpy(
                np.stack([_lr_frame(t - 1), _lr_frame(t), _lr_frame(t + 1)], axis=0)
            )  # (3, 128, 128)
            window_t = lr_t
            window_t1 = torch.from_numpy(
                np.stack([_lr_frame(t), _lr_frame(t + 1), _lr_frame(t + 2)], axis=0)
            )  # (3, 128, 128)
        else:
            lr_t = torch.from_numpy(lr).unsqueeze(0)     # (1, 128, 128)
            window_t = lr_t.repeat(3, 1, 1)
            window_t1 = window_t

        hr_t = torch.from_numpy(hr).unsqueeze(0)  # (1, target_size, target_size)

        if self.augment and random.random() < 0.5:
            lr_t   = torch.flip(lr_t,   dims=[2])
            hr_t   = torch.flip(hr_t,   dims=[2])
            window_t  = torch.flip(window_t,  dims=[2])
            window_t1 = torch.flip(window_t1, dims=[2])

        return {
            "input":     lr_t,
            "target":    hr_t,
            "window_t":  window_t,
            "window_t1": window_t1,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    pixel_criterion: nn.Module,
    temporal_criterion: TemporalConsistencyLoss,
    kspace_criterion: KSpaceConsistencyLoss,
    optimizer: Adam | None,
    device: torch.device,
    max_grad_norm: float,
    lambda_temporal: float,
    lambda_kspace: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {"sup": 0.0, "temporal": 0.0, "kspace": 0.0, "total": 0.0}

    for batch in dataloader:
        inp     = batch["input"].to(device)
        target  = batch["target"].to(device)
        wt      = batch["window_t"].to(device)
        wt1     = batch["window_t1"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            sr = model(inp)
            loss_sup = pixel_criterion(sr, target)

            sr_t  = model(wt)
            sr_t1 = model(wt1)
            loss_temporal = temporal_criterion(sr_t, sr_t1, wt)
            loss_kspace   = kspace_criterion(sr_t, wt[:, 1:2])

            loss_total = loss_sup + lambda_temporal * loss_temporal + lambda_kspace * loss_kspace

            if is_train:
                loss_total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        totals["sup"]      += float(loss_sup.item())
        totals["temporal"] += float(loss_temporal.item())
        totals["kspace"]   += float(loss_kspace.item())
        totals["total"]    += float(loss_total.item())

    n = max(1, len(dataloader))
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    data_dir = Path(args.data_dir)
    max_files = getattr(args, "max_files", None)

    print(f"Loading CMRxRecon dataset from {data_dir} ...", flush=True)
    full_dataset = CMRxReconDataset(
        data_dir=data_dir,
        target_size=args.target_size,
        lr_size=args.lr_size,
        augment=args.augment,
        max_files=max_files,
        in_channels=args.in_channels,
    )

    # 90/10 train/val split
    n_val = max(1, len(full_dataset) // 10)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = build_proposed(
        n_res_blocks=16, n_feats=64, reduction=16, res_scale=0.1,
        in_channels=args.in_channels,
    ).to(device)

    if args.generator_checkpoint:
        ckpt_path = Path(args.generator_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Generator checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded generator from {ckpt_path}", flush=True)

    pixel_criterion   = ForegroundEdgePerceptualLoss(alpha_percep=args.alpha_percep).to(device)
    temporal_criterion = TemporalConsistencyLoss().to(device)
    kspace_criterion  = KSpaceConsistencyLoss(region_size=32)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    best_path = output_dir / "best_model.pth"
    history = []

    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, pixel_criterion, temporal_criterion,
                            kspace_criterion, optimizer, device, args.max_grad_norm,
                            lambda_temporal=args.lambda_temporal,
                            lambda_kspace=args.lambda_kspace)
        val_m   = run_epoch(model, val_loader,   pixel_criterion, temporal_criterion,
                            kspace_criterion, None, device, 0.0,
                            lambda_temporal=args.lambda_temporal,
                            lambda_kspace=args.lambda_kspace)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_m['total']:.6f} (sup={train_m['sup']:.4f} "
            f"t={train_m['temporal']:.4f} ks={train_m['kspace']:.4f}) | "
            f"val={val_m['total']:.6f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s",
            flush=True,
        )
        history.append({"epoch": epoch, "train_total": train_m["total"], "val_total": val_m["total"], **train_m})
        if val_m["total"] < best_val:
            best_val = val_m["total"]
            torch.save({"model_state_dict": model.state_dict()}, best_path)

    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"Pretraining complete. Best checkpoint: {best_path}  (val={best_val:.6f})", flush=True)
    print("Fine-tune on speech MRI using train_gan.py --generator-checkpoint.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain SpeechSR on CMRxRecon cardiac cine MRI")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing CMRxRecon .mat files "
                             "(e.g. MultiCoil/Cine/TrainingSet/FullSample).")
    parser.add_argument("--output-dir", type=str, default="outputs/cmrxrecon_pretrained")
    parser.add_argument("--generator-checkpoint", type=str, default=None,
                        help="Optional checkpoint to warm-start from.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-size", type=int, default=512,
                        help="HR output size (default 512, matches speech MRI pipeline).")
    parser.add_argument("--lr-size", type=int, default=128,
                        help="LR input size via k-space truncation.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of .mat files (e.g. 100 for a quick run).")
    parser.add_argument("--in-channels", type=int, default=3, choices=[1, 3],
                        help="3 = temporal (default), 1 = single-frame.")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--alpha-percep", type=float, default=0.1)
    parser.add_argument("--lambda-temporal", type=float, default=0.10,
                        help="Weight of TemporalConsistencyLoss during pretraining.")
    parser.add_argument("--lambda-kspace", type=float, default=0.05,
                        help="Weight of KSpaceConsistencyLoss during pretraining.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
