import argparse
import itertools
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from models import (
    CombinedL2SSIMLoss,
    ForegroundEdgeLoss,
    ForegroundEdgePerceptualLoss,
    KSpaceConsistencyLoss,
    TemporalConsistencyLoss,
    build_proposed1,
    build_proposed2,
    build_unet,
)

build_proposed = build_proposed2  # default

# Files excluded from dynamic training by default.
_DEFAULT_DYNAMIC_EXCLUDE = {"Subject0021_speech_128x128x200.nii"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_subject_id(name: str) -> str | None:
    match = re.search(r"Subject(\d{4})", name, flags=re.IGNORECASE)
    return match.group(1) if match else None


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v, max_v = float(arr.min()), float(arr.max())
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def load_nifti_frames(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f"Only 2D/3D NIfTI is supported. Got shape {data.shape} for {path}")
    return data


def kspace_zeropad(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape
    tensor = torch.from_numpy(image)
    kspace = torch.fft.fftshift(torch.fft.fft2(tensor))
    padded = torch.zeros((target_size, target_size), dtype=torch.complex64)
    rs = (target_size - h) // 2
    cs = (target_size - w) // 2
    padded[rs:rs + h, cs:cs + w] = kspace
    output = torch.fft.ifft2(torch.fft.ifftshift(padded)).abs().cpu().numpy()
    return normalize01(output)


# ---------------------------------------------------------------------------
# Paired static dataset (supervised anchor)
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    input_path: Path
    target_path: Path
    frame_idx: int
    subject: str


class PairedMRIDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        target_dir: Path,
        model_name: str,
        subjects: list[str] | None = None,
        proposed_target_size: int = 512,
        patch_size: int | None = None,
        augment: bool = False,
        temporal: bool = False,
    ) -> None:
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.model_name = model_name
        self.proposed_target_size = proposed_target_size
        self.patch_size = patch_size
        self.augment = augment
        self.temporal = temporal   # if True, return 3-channel LR (t-1, t, t+1)
        self.samples = self._build_samples(subjects)
        if not self.samples:
            raise ValueError("No paired samples were found. Check naming and input/target folders.")

    def _build_samples(self, subjects: list[str] | None) -> list[Sample]:
        input_files = sorted(self.input_dir.glob("*.nii")) + sorted(self.input_dir.glob("*.nii.gz"))
        samples: list[Sample] = []

        for input_file in input_files:
            subject = extract_subject_id(input_file.name)
            if subject is None:
                continue
            if subjects is not None and subject not in subjects:
                continue

            target_file = self._find_target(input_file)
            if target_file is None:
                continue

            n_frames = load_nifti_frames(input_file).shape[2]
            for frame_idx in range(n_frames):
                samples.append(Sample(input_file, target_file, frame_idx, subject))

        return samples

    def _find_target(self, input_file: Path) -> Path | None:
        candidates = [
            self.target_dir / input_file.name,
            self.target_dir / input_file.name.replace("LR_", "HR_"),
            self.target_dir / input_file.name.replace("Synth_LR", "HR"),
            self.target_dir / input_file.name.replace("128", "512"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        lr_volume = load_nifti_frames(sample.input_path)
        hr_volume = load_nifti_frames(sample.target_path)

        frame_idx = min(sample.frame_idx, hr_volume.shape[2] - 1)
        hr = normalize01(hr_volume[:, :, frame_idx])

        if self.model_name in ("proposed", "proposed2"):
            hr = kspace_zeropad(hr, target_size=self.proposed_target_size)
        else:
            if hr.shape != lr_volume[:, :, 0].shape:
                hr_tensor = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)
                hr = F.interpolate(hr_tensor, size=lr_volume.shape[:2], mode="bilinear", align_corners=False).squeeze().numpy()

        if self.temporal:
            n_frames = lr_volume.shape[2]
            i = sample.frame_idx
            prev_frame = normalize01(lr_volume[:, :, max(i - 1, 0)])
            curr_frame = normalize01(lr_volume[:, :, i])
            next_frame = normalize01(lr_volume[:, :, min(i + 1, n_frames - 1)])
            lr_t = torch.from_numpy(np.stack([prev_frame, curr_frame, next_frame], axis=0))  # (3, H, W)
        else:
            lr_t = torch.from_numpy(normalize01(lr_volume[:, :, sample.frame_idx])).unsqueeze(0)  # (1, H, W)

        hr_t = torch.from_numpy(hr).unsqueeze(0)  # (1, H, W)

        if self.patch_size is not None:
            lr_t, hr_t = self._random_patch(lr_t, hr_t)

        if self.augment and random.random() < 0.5:
            lr_t = torch.flip(lr_t, dims=[2])
            hr_t = torch.flip(hr_t, dims=[2])

        return {"input": lr_t, "target": hr_t}

    def _random_patch(self, lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, lh, lw = lr.shape
        _, hh, hw = hr.shape
        scale = hh // lh

        ps = self.patch_size
        if lh < ps or lw < ps:
            return lr, hr

        y0_lr = random.randint(0, lh - ps)
        x0_lr = random.randint(0, lw - ps)
        y0_hr, x0_hr = y0_lr * scale, x0_lr * scale
        hr_ps = ps * scale

        return (
            lr[:, y0_lr:y0_lr + ps, x0_lr:x0_lr + ps],
            hr[:, y0_hr:y0_hr + hr_ps, x0_hr:x0_hr + hr_ps],
        )


# ---------------------------------------------------------------------------
# Unpaired real dynamic dataset (self-supervised temporal stream)
# ---------------------------------------------------------------------------

class DynamicMRIDataset(Dataset):
    """
    Unpaired dynamic MRI dataset for self-supervised temporal training.

    Each sample returns two adjacent temporal windows from the same volume:
        window_t  = frames (i-1, i,   i+1) as a 3-channel tensor
        window_t1 = frames (i,   i+1, i+2) as a 3-channel tensor

    The overlap (frames t and t+1 appear in both windows) lets the model be
    penalised for inconsistent SR outputs on the shared content.

    Files in the exclude set are skipped (default: the mixed-phoneme speech volume
    which is reserved for held-out qualitative evaluation).
    """

    def __init__(
        self,
        dynamic_dir: Path,
        exclude_files: set[str] | None = None,
        augment: bool = False,
    ) -> None:
        self.augment = augment
        exclude = exclude_files if exclude_files is not None else _DEFAULT_DYNAMIC_EXCLUDE

        nii_files = sorted(dynamic_dir.glob("*.nii")) + sorted(dynamic_dir.glob("*.nii.gz"))
        nii_files = [f for f in nii_files if f.name not in exclude]
        if not nii_files:
            raise ValueError(f"No dynamic NIfTI files found in {dynamic_dir} (after exclusions).")

        # (file, start_frame_i) where window_t uses [i-1, i, i+1] and
        # window_t1 uses [i, i+1, i+2] — so i must be in [1, n_frames-3].
        self.samples: list[tuple[Path, int]] = []
        for f in nii_files:
            n = load_nifti_frames(f).shape[2]
            for i in range(1, n - 2):
                self.samples.append((f, i))

        if not self.samples:
            raise ValueError("No valid dynamic samples found (all volumes have < 4 frames?).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, i = self.samples[idx]
        vol = load_nifti_frames(path)

        def _frame(fi: int) -> np.ndarray:
            return normalize01(vol[:, :, fi])

        # window_t: (i-1, i, i+1)
        w_t = torch.from_numpy(np.stack([_frame(i - 1), _frame(i), _frame(i + 1)], axis=0))
        # window_t1: (i, i+1, i+2)
        w_t1 = torch.from_numpy(np.stack([_frame(i), _frame(i + 1), _frame(i + 2)], axis=0))

        if self.augment and random.random() < 0.5:
            w_t = torch.flip(w_t, dims=[2])
            w_t1 = torch.flip(w_t1, dims=[2])

        return {"window_t": w_t, "window_t1": w_t1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_subjects(all_subjects: list[str], val_subject: str, test_subject: str) -> tuple[list[str], list[str], list[str]]:
    train_subjects = [subj for subj in all_subjects if subj not in {val_subject, test_subject}]
    if not train_subjects:
        raise ValueError("No training subjects left after val/test split. Provide more subject IDs.")
    return train_subjects, [val_subject], [test_subject]


def build_model(model_name: str, hyperparams: dict[str, Any], temporal: bool = False) -> nn.Module:
    in_channels = 3 if temporal else 1

    if model_name == "unet":
        return build_unet(
            base_filters=int(hyperparams.get("base_filters", 32)),
            bilinear=bool(hyperparams.get("bilinear", True)),
        )

    if model_name == "proposed1":
        return build_proposed1(
            n_res_blocks=int(hyperparams.get("n_res_blocks", 16)),
            n_feats=int(hyperparams.get("n_feats", 64)),
            reduction=int(hyperparams.get("reduction", 16)),
            res_scale=float(hyperparams.get("res_scale", 0.1)),
        )

    if model_name in ("proposed", "proposed2"):
        return build_proposed2(
            n_res_blocks=int(hyperparams.get("n_res_blocks", 16)),
            n_feats=int(hyperparams.get("n_feats", 64)),
            reduction=int(hyperparams.get("reduction", 16)),
            res_scale=float(hyperparams.get("res_scale", 0.1)),
            in_channels=in_channels,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def build_loss(model_name: str, hyperparams: dict[str, Any], perceptual: bool = False) -> nn.Module:
    if model_name == "unet":
        return CombinedL2SSIMLoss(alpha_l2=float(hyperparams.get("alpha_l2", 0.7)))
    if perceptual:
        return ForegroundEdgePerceptualLoss(
            alpha_l1=float(hyperparams.get("alpha_l1", 0.25)),
            alpha_sobel=float(hyperparams.get("alpha_sobel", 0.35)),
            alpha_laplacian=float(hyperparams.get("alpha_laplacian", 0.25)),
            alpha_ssim=float(hyperparams.get("alpha_ssim", 0.15)),
            alpha_percep=float(hyperparams.get("alpha_percep", 0.1)),
        )
    return ForegroundEdgeLoss(
        alpha_l1=float(hyperparams.get("alpha_l1", 0.25)),
        alpha_sobel=float(hyperparams.get("alpha_sobel", 0.35)),
        alpha_laplacian=float(hyperparams.get("alpha_laplacian", 0.25)),
        alpha_ssim=float(hyperparams.get("alpha_ssim", 0.15)),
    )


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Supervised-only epoch (used for val/test)."""
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(1, len(dataloader))


def run_joint_train_epoch(
    model: nn.Module,
    paired_loader: DataLoader,
    dynamic_loader: DataLoader | None,
    criterion: nn.Module,
    temporal_criterion: TemporalConsistencyLoss,
    kspace_criterion: KSpaceConsistencyLoss,
    optimizer: Adam,
    device: torch.device,
    max_grad_norm: float,
    lambda_temporal: float,
    lambda_kspace_dyn: float,
) -> dict[str, float]:
    """
    Joint supervised + self-supervised training epoch.

    For each paired (LR, HR) batch, we also draw a dynamic batch
    (window_t, window_t1) and add temporal consistency + k-space
    consistency terms. When dynamic_loader is None, this degrades to
    a standard supervised epoch.
    """
    model.train()
    totals: dict[str, float] = {"sup": 0.0, "temporal": 0.0, "kspace_dyn": 0.0, "total": 0.0}

    dyn_iter = itertools.cycle(dynamic_loader) if dynamic_loader is not None else None

    for batch in paired_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Supervised loss (paired static)
        sr = model(inputs)
        loss_sup = criterion(sr, targets)

        loss_total = loss_sup
        loss_temporal = torch.zeros(1, device=device)
        loss_kspace_dyn = torch.zeros(1, device=device)

        # Auxiliary dynamic stream
        if dyn_iter is not None:
            dyn_batch = next(dyn_iter)
            wt = dyn_batch["window_t"].to(device)    # (B, 3, H, W)
            wt1 = dyn_batch["window_t1"].to(device)  # (B, 3, H, W)

            sr_t = model(wt)
            sr_t1 = model(wt1)

            loss_temporal = temporal_criterion(sr_t, sr_t1, wt)
            # K-space consistency: SR vs centre frame of the LR window
            loss_kspace_dyn = kspace_criterion(sr_t, wt[:, 1:2])

            loss_total = loss_sup + lambda_temporal * loss_temporal + lambda_kspace_dyn * loss_kspace_dyn

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        totals["sup"] += float(loss_sup.item())
        totals["temporal"] += float(loss_temporal.item())
        totals["kspace_dyn"] += float(loss_kspace_dyn.item())
        totals["total"] += float(loss_total.item())

    n = max(1, len(paired_loader))
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_once(args: argparse.Namespace, hyperparams: dict[str, Any], run_name: str) -> float:
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    temporal = getattr(args, "temporal", False)
    use_dynamic = getattr(args, "dynamic_dir", None) and not getattr(args, "no_dynamic", False)
    lambda_temporal = float(getattr(args, "lambda_temporal", 0.10))
    lambda_kspace_dyn = float(getattr(args, "lambda_kspace_dyn", 0.05))

    train_subjects, val_subjects, test_subjects = split_subjects(args.subjects, args.val_subject, args.test_subject)

    patch_size = args.patch_size if hasattr(args, "patch_size") else None
    augment = args.augment if hasattr(args, "augment") else False

    train_dataset = PairedMRIDataset(
        input_dir=Path(args.input_dir),
        target_dir=Path(args.target_dir),
        model_name=args.model,
        subjects=train_subjects,
        proposed_target_size=args.proposed_target_size,
        patch_size=patch_size,
        augment=augment,
        temporal=temporal,
    )
    val_dataset = PairedMRIDataset(
        input_dir=Path(args.input_dir),
        target_dir=Path(args.target_dir),
        model_name=args.model,
        subjects=val_subjects,
        proposed_target_size=args.proposed_target_size,
        temporal=temporal,
    )
    test_dataset = PairedMRIDataset(
        input_dir=Path(args.input_dir),
        target_dir=Path(args.target_dir),
        model_name=args.model,
        subjects=test_subjects,
        proposed_target_size=args.proposed_target_size,
        temporal=temporal,
    )

    batch_size = int(hyperparams.get("batch_size", args.batch_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    dynamic_loader = None
    if use_dynamic:
        exclude = set(getattr(args, "exclude_files", [])) | _DEFAULT_DYNAMIC_EXCLUDE
        dyn_dataset = DynamicMRIDataset(
            dynamic_dir=Path(args.dynamic_dir),
            exclude_files=exclude,
            augment=augment,
        )
        dynamic_loader = DataLoader(dyn_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        print(f"Dynamic stream: {len(dyn_dataset)} samples from {args.dynamic_dir}", flush=True)

    model = build_model(args.model, hyperparams, temporal=temporal).to(device)
    perceptual = getattr(args, "perceptual_loss", False)
    criterion = build_loss(args.model, hyperparams, perceptual=perceptual).to(device)
    temporal_criterion = TemporalConsistencyLoss().to(device)
    kspace_criterion = KSpaceConsistencyLoss(region_size=32)

    optimizer = Adam(
        model.parameters(),
        lr=float(hyperparams.get("lr", args.lr)),
        weight_decay=float(hyperparams.get("weight_decay", args.weight_decay)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=float(hyperparams.get("eta_min", 1e-6)))

    best_val_loss = float("inf")
    best_path = output_dir / "best_model.pth"

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_joint_train_epoch(
            model, train_loader, dynamic_loader, criterion,
            temporal_criterion, kspace_criterion,
            optimizer, device,
            max_grad_norm=float(hyperparams.get("max_grad_norm", args.max_grad_norm)),
            lambda_temporal=lambda_temporal,
            lambda_kspace_dyn=lambda_kspace_dyn,
        )
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device, max_grad_norm=0.0)
        scheduler.step()

        elapsed = time.time() - start
        dyn_str = (f" t={train_metrics['temporal']:.4f} ks={train_metrics['kspace_dyn']:.4f}"
                   if use_dynamic else "")
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_metrics['total']:.6f}{dyn_str} | "
            f"val={val_loss:.6f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s",
            flush=True,
        )

        history.append({"epoch": epoch, "train_loss": train_metrics["total"], "val_loss": val_loss, **train_metrics})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "hyperparams": hyperparams}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = run_epoch(model, test_loader, criterion, optimizer=None, device=device, max_grad_norm=0.0)

    metrics = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_test_samples": len(test_dataset),
        "hyperparams": hyperparams,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_val_loss


def run_optuna(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is not installed. Install dependencies from requirements.txt first.") from exc

    def suggest_hyperparams(trial: "optuna.Trial") -> dict[str, Any]:
        params: dict[str, Any] = {
            "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-4),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
            "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0]),
        }
        if args.model == "unet":
            params.update({"base_filters": trial.suggest_categorical("base_filters", [16, 32, 64]), "alpha_l2": trial.suggest_float("alpha_l2", 0.5, 0.9)})
        else:
            params.update({
                "n_res_blocks": trial.suggest_categorical("n_res_blocks", [8, 12, 16]),
                "n_feats": trial.suggest_categorical("n_feats", [32, 48, 64]),
                "reduction": trial.suggest_categorical("reduction", [8, 16]),
                "res_scale": trial.suggest_categorical("res_scale", [0.05, 0.1, 0.2]),
                "alpha_l1": trial.suggest_float("alpha_l1", 0.2, 0.4),
                "alpha_sobel": trial.suggest_float("alpha_sobel", 0.2, 0.45),
                "alpha_laplacian": trial.suggest_float("alpha_laplacian", 0.1, 0.35),
                "alpha_ssim": trial.suggest_float("alpha_ssim", 0.1, 0.25),
            })
            if getattr(args, "perceptual_loss", False):
                params["alpha_percep"] = trial.suggest_float("alpha_percep", 0.05, 0.2)
        return params

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_hyperparams(trial)
        original_epochs = args.epochs
        args.epochs = args.hpo_epochs
        run_name = f"optuna_trial_{trial.number:03d}"
        val_loss = train_once(args, params, run_name=run_name)
        args.epochs = original_epochs
        return val_loss

    storage = f"sqlite:///{Path(args.output_dir) / 'optuna_study.db'}"
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{args.model}_hpo",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial.params
    with (Path(args.output_dir) / "best_hyperparameters.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Best trial {study.best_trial.number} | value={study.best_trial.value:.6f}")
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpeechSR with optional dynamic auxiliary stream")
    parser.add_argument("--model", choices=["unet", "proposed1", "proposed2", "proposed"], required=True)
    parser.add_argument("--input-dir", type=str, default="data/images/Synth_LR")
    parser.add_argument("--target-dir", type=str, default="data/images/HR")
    parser.add_argument("--output-dir", type=str, default="outputs")

    parser.add_argument("--subjects", nargs="+", default=["0021", "0022", "0023", "0024", "0025", "0026", "0027"])
    parser.add_argument("--val-subject", type=str, default="0022")
    parser.add_argument("--test-subject", type=str, default="0021")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--proposed-target-size", type=int, default=512,
                        help="HR output size in pixels (default 512 = 4× from 128 LR).")

    parser.add_argument("--patch-size", type=int, default=None, metavar="N")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--perceptual-loss", action="store_true")

    # Temporal mode
    parser.add_argument("--temporal", action="store_true",
                        help="Stack (t-1, t, t+1) as 3-channel input (required for dynamic SR).")

    # Dynamic auxiliary stream
    parser.add_argument("--dynamic-dir", type=str, default=None,
                        help="Path to Dynamic_128/ directory of real unpaired dynamic NIfTI files.")
    parser.add_argument("--lambda-temporal", type=float, default=0.10,
                        help="Weight of motion-weighted temporal consistency loss on dynamic stream.")
    parser.add_argument("--lambda-kspace-dyn", type=float, default=0.05,
                        help="Weight of k-space data consistency loss on dynamic stream.")
    parser.add_argument("--exclude-files", nargs="*", default=[],
                        help="Additional dynamic filenames to exclude from training.")
    parser.add_argument("--no-dynamic", action="store_true",
                        help="Disable dynamic stream even if --dynamic-dir is set (ablation).")

    # Optuna
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--hpo-epochs", type=int, default=20)
    parser.add_argument("--train-after-hpo", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.use_optuna:
        best_params = run_optuna(args)
        if args.train_after_hpo:
            train_once(args, best_params, run_name="best_from_optuna")
    else:
        train_once(args, hyperparams={}, run_name=f"{args.model}_run")


if __name__ == "__main__":
    main()
