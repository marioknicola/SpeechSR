"""
Two-stage GAN training for SpeechSR.

Stage 1 — generator pretraining:
    Trains the generator with ForegroundEdgePerceptualLoss + optional dynamic
    auxiliary losses (temporal consistency + k-space data consistency).

Stage 2 — adversarial fine-tuning:
    Adds a PatchGAN discriminator with relativistic adversarial loss. Dynamic
    SR batches are fed to the discriminator alongside synthetic batches so the
    generator is encouraged to produce outputs with consistent statistics across
    both domains.

Usage:
    # Full two-stage with dynamic stream
    python train_gan.py \\
        --input-dir data/images/Synth_LR \\
        --target-dir data/images/HR \\
        --dynamic-dir data/images/Dynamic_128 \\
        --output-dir outputs/gan \\
        --pretrain-epochs 150 --gan-epochs 0 \\
        --temporal

    # Skip pretraining — GAN fine-tune from existing generator
    python train_gan.py \\
        --input-dir data/images/Synth_LR \\
        --target-dir data/images/HR \\
        --output-dir outputs/gan \\
        --pretrain-epochs 0 --gan-epochs 50 \\
        --lambda-adv 0.003 \\
        --generator-checkpoint outputs/pretrained_generator.pth \\
        --temporal
"""

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models import (
    ForegroundEdgePerceptualLoss,
    KSpaceConsistencyLoss,
    TemporalConsistencyLoss,
    build_discriminator,
    build_proposed2,
)
from train import (
    DynamicMRIDataset,
    PairedMRIDataset,
    _DEFAULT_DYNAMIC_EXCLUDE,
    pick_device,
    set_seed,
    split_subjects,
)


# ---------------------------------------------------------------------------
# Adversarial losses (Relativistic average GAN)
# ---------------------------------------------------------------------------

def generator_adv_loss(real_out: torch.Tensor, fake_out: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        fake_out - real_out.mean(),
        torch.ones_like(fake_out),
    )


def discriminator_adv_loss(real_out: torch.Tensor, fake_out: torch.Tensor) -> torch.Tensor:
    real_loss = F.binary_cross_entropy_with_logits(
        real_out - fake_out.mean(),
        torch.ones_like(real_out),
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_out - real_out.mean(),
        torch.zeros_like(fake_out),
    )
    return (real_loss + fake_loss) / 2


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def pretrain_epoch(
    generator: nn.Module,
    paired_loader: DataLoader,
    dynamic_loader: DataLoader | None,
    pixel_criterion: nn.Module,
    temporal_criterion: TemporalConsistencyLoss,
    kspace_criterion: KSpaceConsistencyLoss,
    optimizer: Adam,
    device: torch.device,
    max_grad_norm: float,
    lambda_temporal: float,
    lambda_kspace_dyn: float,
) -> dict[str, float]:
    generator.train()
    totals: dict[str, float] = {"sup": 0.0, "temporal": 0.0, "kspace_dyn": 0.0, "total": 0.0}

    dyn_iter = itertools.cycle(dynamic_loader) if dynamic_loader is not None else None

    for batch in paired_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        sr = generator(inputs)
        loss_sup = pixel_criterion(sr, targets)

        loss_temporal = torch.zeros(1, device=device)
        loss_kspace_dyn = torch.zeros(1, device=device)
        loss_total = loss_sup

        if dyn_iter is not None:
            dyn_batch = next(dyn_iter)
            wt = dyn_batch["window_t"].to(device)
            wt1 = dyn_batch["window_t1"].to(device)

            sr_t = generator(wt)
            sr_t1 = generator(wt1)

            loss_temporal = temporal_criterion(sr_t, sr_t1, wt)
            loss_kspace_dyn = kspace_criterion(sr_t, wt[:, 1:2])
            loss_total = loss_sup + lambda_temporal * loss_temporal + lambda_kspace_dyn * loss_kspace_dyn

        loss_total.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        totals["sup"] += float(loss_sup.item())
        totals["temporal"] += float(loss_temporal.item())
        totals["kspace_dyn"] += float(loss_kspace_dyn.item())
        totals["total"] += float(loss_total.item())

    n = max(1, len(paired_loader))
    return {k: v / n for k, v in totals.items()}


def gan_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    paired_loader: DataLoader,
    dynamic_loader: DataLoader | None,
    pixel_criterion: nn.Module,
    kspace_criterion: KSpaceConsistencyLoss,
    temporal_criterion: TemporalConsistencyLoss,
    g_optimizer: Adam,
    d_optimizer: Adam,
    device: torch.device,
    max_grad_norm: float,
    lambda_adv: float,
    lambda_kspace: float,
    lambda_temporal: float,
    lambda_kspace_dyn: float,
) -> dict[str, float]:
    generator.train()
    discriminator.train()

    totals: dict[str, float] = {
        "g_total": 0.0, "g_pixel": 0.0, "g_adv": 0.0,
        "g_kspace": 0.0, "g_temporal": 0.0, "d_loss": 0.0,
    }

    dyn_iter = itertools.cycle(dynamic_loader) if dynamic_loader is not None else None

    for batch in paired_loader:
        lr = batch["input"].to(device)
        hr = batch["target"].to(device)

        # Optional dynamic batch
        dyn_sr = None
        if dyn_iter is not None:
            dyn_batch = next(dyn_iter)
            wt = dyn_batch["window_t"].to(device)
            wt1 = dyn_batch["window_t1"].to(device)

        # ------------------------------------------------------------------
        # Discriminator step
        # ------------------------------------------------------------------
        with torch.no_grad():
            sr = generator(lr)
            if dyn_iter is not None:
                dyn_sr = generator(wt)

        real_out = discriminator(hr)
        if dyn_iter is not None:
            # Feed both synthetic SR and dynamic SR as fake samples
            fake_combined = torch.cat([sr, dyn_sr], dim=0)
            real_expanded = real_out.repeat(2, 1, 1, 1) if real_out.shape[0] == sr.shape[0] else real_out
            fake_out = discriminator(fake_combined.detach())
            d_loss = discriminator_adv_loss(real_expanded, fake_out)
        else:
            fake_out = discriminator(sr.detach())
            d_loss = discriminator_adv_loss(real_out, fake_out)

        d_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
        d_optimizer.step()

        # ------------------------------------------------------------------
        # Generator step
        # ------------------------------------------------------------------
        sr = generator(lr)
        real_out = discriminator(hr).detach()
        fake_out = discriminator(sr)

        g_pixel = pixel_criterion(sr, hr)
        g_adv = generator_adv_loss(real_out, fake_out)
        g_kspace = kspace_criterion(sr, lr)

        g_temporal = torch.zeros(1, device=device)
        g_kspace_dyn = torch.zeros(1, device=device)
        if dyn_iter is not None:
            sr_t = generator(wt)
            sr_t1 = generator(wt1)
            g_temporal = temporal_criterion(sr_t, sr_t1, wt)
            g_kspace_dyn = kspace_criterion(sr_t, wt[:, 1:2])

        g_loss = (
            g_pixel
            + lambda_adv * g_adv
            + lambda_kspace * g_kspace
            + lambda_temporal * g_temporal
            + lambda_kspace_dyn * g_kspace_dyn
        )

        g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
        g_optimizer.step()

        totals["g_total"] += float(g_loss.item())
        totals["g_pixel"] += float(g_pixel.item())
        totals["g_adv"] += float(g_adv.item())
        totals["g_kspace"] += float(g_kspace.item())
        totals["g_temporal"] += float(g_temporal.item())
        totals["d_loss"] += float(d_loss.item())

    n = max(1, len(paired_loader))
    return {k: v / n for k, v in totals.items()}


def compute_val_loss(
    generator: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    generator.eval()
    total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            total += float(criterion(generator(inputs), targets).item())
    return total / max(1, len(dataloader))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    temporal = getattr(args, "temporal", False)
    use_dynamic = getattr(args, "dynamic_dir", None) and not getattr(args, "no_dynamic", False)
    lambda_temporal = float(getattr(args, "lambda_temporal", 0.10))
    lambda_kspace_dyn = float(getattr(args, "lambda_kspace_dyn", 0.05))

    train_subjects, val_subjects, _ = split_subjects(args.subjects, args.val_subject, args.test_subject)

    common_dataset_kwargs: dict[str, Any] = dict(
        input_dir=Path(args.input_dir),
        target_dir=Path(args.target_dir),
        model_name="proposed2",
        proposed_target_size=args.proposed_target_size,
        temporal=temporal,
    )
    train_dataset = PairedMRIDataset(
        **common_dataset_kwargs,
        subjects=train_subjects,
        patch_size=args.patch_size,
        augment=args.augment,
    )
    val_dataset = PairedMRIDataset(**common_dataset_kwargs, subjects=val_subjects)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    dynamic_loader = None
    if use_dynamic:
        exclude = set(getattr(args, "exclude_files", [])) | _DEFAULT_DYNAMIC_EXCLUDE
        dyn_dataset = DynamicMRIDataset(
            dynamic_dir=Path(args.dynamic_dir),
            exclude_files=exclude,
            augment=args.augment,
        )
        dynamic_loader = DataLoader(dyn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        print(f"Dynamic stream: {len(dyn_dataset)} samples from {args.dynamic_dir}", flush=True)

    in_channels = 3 if temporal else 1
    generator = build_proposed2(n_res_blocks=16, n_feats=64, reduction=16, res_scale=0.1, in_channels=in_channels).to(device)
    discriminator = build_discriminator(base_filters=64).to(device)

    pixel_criterion = ForegroundEdgePerceptualLoss(alpha_percep=args.alpha_percep).to(device)
    kspace_criterion = KSpaceConsistencyLoss(region_size=32)
    temporal_criterion = TemporalConsistencyLoss().to(device)

    # -----------------------------------------------------------------------
    # Stage 1: Generator pretraining
    # -----------------------------------------------------------------------
    pretrain_ckpt = output_dir / "pretrained_generator.pth"

    if args.pretrain_epochs > 0:
        if args.generator_checkpoint:
            print(f"Loading generator from {args.generator_checkpoint} for warmstart.", flush=True)
            _load_generator(generator, Path(args.generator_checkpoint), device)

        g_optimizer = Adam(generator.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        g_scheduler = CosineAnnealingLR(g_optimizer, T_max=args.pretrain_epochs, eta_min=1e-6)

        best_val = float("inf")
        pretrain_history = []
        print(f"\n--- Stage 1: Pretraining generator for {args.pretrain_epochs} epochs ---", flush=True)
        for epoch in range(1, args.pretrain_epochs + 1):
            t0 = time.time()
            metrics = pretrain_epoch(
                generator, train_loader, dynamic_loader,
                pixel_criterion, temporal_criterion, kspace_criterion,
                g_optimizer, device, args.max_grad_norm,
                lambda_temporal=lambda_temporal,
                lambda_kspace_dyn=lambda_kspace_dyn,
            )
            val_l = compute_val_loss(generator, val_loader, pixel_criterion, device)
            g_scheduler.step()
            elapsed = time.time() - t0
            dyn_str = (f" t={metrics['temporal']:.4f} ks={metrics['kspace_dyn']:.4f}"
                       if use_dynamic else "")
            print(
                f"Pretrain {epoch:03d}/{args.pretrain_epochs} | "
                f"train={metrics['total']:.6f}{dyn_str} | val={val_l:.6f} | {elapsed:.1f}s",
                flush=True,
            )
            pretrain_history.append({"epoch": epoch, "val_loss": val_l, **metrics})
            if val_l < best_val:
                best_val = val_l
                torch.save({"model_state_dict": generator.state_dict()}, pretrain_ckpt)

        with (output_dir / "pretrain_history.json").open("w") as f:
            json.dump(pretrain_history, f, indent=2)
        print(f"Saved pretrained generator: {pretrain_ckpt}", flush=True)

    elif args.generator_checkpoint:
        print(f"Loading generator from {args.generator_checkpoint} (skipping pretraining).", flush=True)
        _load_generator(generator, Path(args.generator_checkpoint), device)
        torch.save({"model_state_dict": generator.state_dict()}, pretrain_ckpt)

    # -----------------------------------------------------------------------
    # Stage 2: Adversarial fine-tuning
    # -----------------------------------------------------------------------
    if args.gan_epochs <= 0:
        print("GAN epochs = 0, stopping after pretraining.", flush=True)
        return

    if pretrain_ckpt.exists():
        _load_generator(generator, pretrain_ckpt, device)

    g_optimizer = Adam(generator.parameters(), lr=args.gan_lr, weight_decay=args.weight_decay)
    d_optimizer = Adam(discriminator.parameters(), lr=args.gan_lr * 0.5, weight_decay=args.weight_decay)
    g_scheduler = CosineAnnealingLR(g_optimizer, T_max=args.gan_epochs, eta_min=1e-7)
    d_scheduler = CosineAnnealingLR(d_optimizer, T_max=args.gan_epochs, eta_min=1e-7)

    best_val = float("inf")
    gan_history = []
    best_gan_ckpt = output_dir / "best_gan_generator.pth"

    print(f"\n--- Stage 2: GAN fine-tuning for {args.gan_epochs} epochs ---", flush=True)
    print(f"    lambda_adv={args.lambda_adv}  lambda_kspace={args.lambda_kspace}", flush=True)
    if use_dynamic:
        print(f"    lambda_temporal={lambda_temporal}  lambda_kspace_dyn={lambda_kspace_dyn}", flush=True)

    for epoch in range(1, args.gan_epochs + 1):
        t0 = time.time()
        metrics = gan_epoch(
            generator, discriminator,
            train_loader, dynamic_loader,
            pixel_criterion, kspace_criterion, temporal_criterion,
            g_optimizer, d_optimizer,
            device, args.max_grad_norm,
            lambda_adv=args.lambda_adv,
            lambda_kspace=args.lambda_kspace,
            lambda_temporal=lambda_temporal,
            lambda_kspace_dyn=lambda_kspace_dyn,
        )
        val_l = compute_val_loss(generator, val_loader, pixel_criterion, device)
        g_scheduler.step()
        d_scheduler.step()
        elapsed = time.time() - t0

        print(
            f"GAN {epoch:03d}/{args.gan_epochs} | "
            f"G={metrics['g_total']:.4f} (pix={metrics['g_pixel']:.4f} "
            f"adv={metrics['g_adv']:.4f} ks={metrics['g_kspace']:.4f}"
            + (f" t={metrics['g_temporal']:.4f}" if use_dynamic else "")
            + f") | D={metrics['d_loss']:.4f} | val={val_l:.6f} | {elapsed:.1f}s",
            flush=True,
        )
        gan_history.append({"epoch": epoch, "val_loss": val_l, **metrics})

        if val_l < best_val:
            best_val = val_l
            torch.save({"model_state_dict": generator.state_dict()}, best_gan_ckpt)

        torch.save({"model_state_dict": discriminator.state_dict()}, output_dir / "discriminator_latest.pth")

    with (output_dir / "gan_history.json").open("w") as f:
        json.dump(gan_history, f, indent=2)
    print(f"Saved best GAN generator: {best_gan_ckpt}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_generator(model: nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage GAN training for SpeechSR")

    # Data
    parser.add_argument("--input-dir", type=str, default="data/images/Synth_LR")
    parser.add_argument("--target-dir", type=str, default="data/images/HR")
    parser.add_argument("--output-dir", type=str, default="outputs/gan")
    parser.add_argument("--subjects", nargs="+", default=["0021", "0022", "0023", "0024", "0025", "0026", "0027"])
    parser.add_argument("--val-subject", type=str, default="0022")
    parser.add_argument("--test-subject", type=str, default="0021")
    parser.add_argument("--proposed-target-size", type=int, default=512)

    # Patching and augmentation
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--augment", action="store_true", default=True)

    # Temporal mode
    parser.add_argument("--temporal", action="store_true",
                        help="3-channel input (t-1, t, t+1). Required when using dynamic stream.")

    # Dynamic auxiliary stream
    parser.add_argument("--dynamic-dir", type=str, default=None)
    parser.add_argument("--lambda-temporal", type=float, default=0.10)
    parser.add_argument("--lambda-kspace-dyn", type=float, default=0.05)
    parser.add_argument("--exclude-files", nargs="*", default=[])
    parser.add_argument("--no-dynamic", action="store_true")

    # Stage 1 — pretraining
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--generator-checkpoint", type=str, default=None)

    # Stage 2 — GAN
    parser.add_argument("--gan-epochs", type=int, default=200)
    parser.add_argument("--gan-lr", type=float, default=1e-4)
    parser.add_argument("--lambda-adv", type=float, default=0.003)
    parser.add_argument("--lambda-kspace", type=float, default=0.1)

    # Loss
    parser.add_argument("--alpha-percep", type=float, default=0.1)

    # Optimisation
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.patch_size == 0:
        args.patch_size = None
    train(args)


if __name__ == "__main__":
    main()
