"""
Two-stage GAN training for SpeechSR.

Stage 1 — generator pretraining:
    Trains the generator with ForegroundEdgePerceptualLoss only. This gives
    the generator a strong starting point so the discriminator does not win
    immediately in Stage 2. Skip this stage (--pretrain-epochs 0) if you
    already have a good generator checkpoint from train.py.

Stage 2 — adversarial fine-tuning:
    Adds a PatchGAN discriminator with relativistic adversarial loss. A
    k-space consistency term prevents the GAN from hallucinating anatomy by
    enforcing that the low-frequency content of the SR output matches the
    LR input in a 32×32 region around the DC peak.

Usage:
    # Full two-stage run
    python train_gan.py \\
        --input-dir data/images/Synth_LR \\
        --target-dir data/images/HR \\
        --output-dir outputs/gan \\
        --pretrain-epochs 100 --gan-epochs 200

    # Skip pretraining — fine-tune an existing generator
    python train_gan.py \\
        --input-dir data/images/Synth_LR \\
        --target-dir data/images/HR \\
        --output-dir outputs/gan \\
        --pretrain-epochs 0 --gan-epochs 200 \\
        --generator-checkpoint outputs/proposed_run/best_model.pth

    # Stage 1 only (useful to compare with/without GAN)
    python train_gan.py \\
        --input-dir data/images/Synth_LR \\
        --target-dir data/images/HR \\
        --output-dir outputs/gan \\
        --pretrain-epochs 100 --gan-epochs 0
"""

import argparse
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
    build_discriminator,
    build_proposed2,
)
from train import (
    PairedMRIDataset,
    pick_device,
    set_seed,
    split_subjects,
)


# ---------------------------------------------------------------------------
# Adversarial losses
# ---------------------------------------------------------------------------

def generator_adv_loss(real_out: torch.Tensor, fake_out: torch.Tensor) -> torch.Tensor:
    """
    Relativistic average GAN loss for the generator.
    Makes fake samples look more real than real samples on average.
    More stable than vanilla BCE for SR tasks (used in ESRGAN).
    """
    return F.binary_cross_entropy_with_logits(
        fake_out - real_out.mean(),
        torch.ones_like(fake_out),
    )


def discriminator_adv_loss(real_out: torch.Tensor, fake_out: torch.Tensor) -> torch.Tensor:
    """Relativistic average GAN loss for the discriminator."""
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
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    generator.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        sr = generator(inputs)
        loss = criterion(sr, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        running_loss += float(loss.item())
    return running_loss / max(1, len(dataloader))


def gan_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    pixel_criterion: nn.Module,
    kspace_criterion: KSpaceConsistencyLoss,
    g_optimizer: Adam,
    d_optimizer: Adam,
    device: torch.device,
    max_grad_norm: float,
    lambda_adv: float,
    lambda_kspace: float,
) -> dict[str, float]:
    generator.train()
    discriminator.train()

    totals: dict[str, float] = {"g_total": 0.0, "g_pixel": 0.0, "g_adv": 0.0, "g_kspace": 0.0, "d_loss": 0.0}

    for batch in dataloader:
        lr = batch["input"].to(device)
        hr = batch["target"].to(device)

        # ------------------------------------------------------------------
        # Discriminator step
        # ------------------------------------------------------------------
        with torch.no_grad():
            sr = generator(lr)

        real_out = discriminator(hr)
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

        g_loss = g_pixel + lambda_adv * g_adv + lambda_kspace * g_kspace

        g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
        g_optimizer.step()

        totals["g_total"] += float(g_loss.item())
        totals["g_pixel"] += float(g_pixel.item())
        totals["g_adv"] += float(g_adv.item())
        totals["g_kspace"] += float(g_kspace.item())
        totals["d_loss"] += float(d_loss.item())

    n = max(1, len(dataloader))
    return {k: v / n for k, v in totals.items()}


def val_loss(
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

    train_subjects, val_subjects, _ = split_subjects(args.subjects, args.val_subject, args.test_subject)

    common_dataset_kwargs: dict[str, Any] = dict(
        input_dir=Path(args.input_dir),
        target_dir=Path(args.target_dir),
        model_name="proposed2",
        proposed_target_size=args.proposed_target_size,
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

    generator = build_proposed2(n_res_blocks=16, n_feats=64, reduction=16, res_scale=0.1).to(device)
    discriminator = build_discriminator(base_filters=64).to(device)

    pixel_criterion = ForegroundEdgePerceptualLoss(alpha_percep=args.alpha_percep).to(device)
    kspace_criterion = KSpaceConsistencyLoss(region_size=32)

    # -----------------------------------------------------------------------
    # Stage 1: Generator pretraining
    # -----------------------------------------------------------------------
    pretrain_ckpt = output_dir / "pretrained_generator.pth"

    if args.pretrain_epochs > 0:
        if args.generator_checkpoint:
            print(f"Loading generator from {args.generator_checkpoint} for pretraining warmstart.", flush=True)
            _load_generator(generator, Path(args.generator_checkpoint), device)

        g_optimizer = Adam(generator.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        g_scheduler = CosineAnnealingLR(g_optimizer, T_max=args.pretrain_epochs, eta_min=1e-6)

        best_val = float("inf")
        pretrain_history = []
        print(f"\n--- Stage 1: Pretraining generator for {args.pretrain_epochs} epochs ---", flush=True)
        for epoch in range(1, args.pretrain_epochs + 1):
            t0 = time.time()
            train_l = pretrain_epoch(generator, train_loader, pixel_criterion, g_optimizer, device, args.max_grad_norm)
            val_l = val_loss(generator, val_loader, pixel_criterion, device)
            g_scheduler.step()
            elapsed = time.time() - t0
            print(f"Pretrain {epoch:03d}/{args.pretrain_epochs} | train={train_l:.6f} | val={val_l:.6f} | {elapsed:.1f}s", flush=True)
            pretrain_history.append({"epoch": epoch, "train_loss": train_l, "val_loss": val_l})
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

    # Reload best pretrained generator before GAN stage
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
    for epoch in range(1, args.gan_epochs + 1):
        t0 = time.time()
        metrics = gan_epoch(
            generator, discriminator,
            train_loader, pixel_criterion, kspace_criterion,
            g_optimizer, d_optimizer,
            device, args.max_grad_norm,
            lambda_adv=args.lambda_adv,
            lambda_kspace=args.lambda_kspace,
        )
        val_l = val_loss(generator, val_loader, pixel_criterion, device)
        g_scheduler.step()
        d_scheduler.step()
        elapsed = time.time() - t0

        print(
            f"GAN {epoch:03d}/{args.gan_epochs} | "
            f"G={metrics['g_total']:.4f} (pix={metrics['g_pixel']:.4f} "
            f"adv={metrics['g_adv']:.4f} ks={metrics['g_kspace']:.4f}) | "
            f"D={metrics['d_loss']:.4f} | val={val_l:.6f} | {elapsed:.1f}s",
            flush=True,
        )
        gan_history.append({"epoch": epoch, "val_loss": val_l, **metrics})

        if val_l < best_val:
            best_val = val_l
            torch.save({"model_state_dict": generator.state_dict()}, best_gan_ckpt)

        # Save discriminator separately in case fine-tuning needs to be resumed
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
    parser.add_argument("--proposed-target-size", type=int, default=1024)

    # Patching and augmentation
    parser.add_argument("--patch-size", type=int, default=64,
                        help="LR patch size (default 64 → 512 HR patch). Set to 0 to disable patching.")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Random horizontal flip (default: on).")

    # Stage 1 — pretraining
    parser.add_argument("--pretrain-epochs", type=int, default=100,
                        help="Generator pretraining epochs before GAN (0 to skip).")
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--generator-checkpoint", type=str, default=None,
                        help="Path to existing generator .pth to warm-start from.")

    # Stage 2 — GAN
    parser.add_argument("--gan-epochs", type=int, default=200,
                        help="Adversarial fine-tuning epochs (0 to stop after pretraining).")
    parser.add_argument("--gan-lr", type=float, default=1e-4,
                        help="Generator LR for GAN stage. Discriminator uses half this value.")
    parser.add_argument("--lambda-adv", type=float, default=0.01,
                        help="Weight of the adversarial loss. Start small (0.01) and increase if output is still blurry.")
    parser.add_argument("--lambda-kspace", type=float, default=0.1,
                        help="Weight of the k-space consistency loss. Prevents hallucination of anatomy.")

    # Loss
    parser.add_argument("--alpha-percep", type=float, default=0.1,
                        help="Weight of VGG perceptual term in ForegroundEdgePerceptualLoss.")

    # Optimisation
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    # patch_size=0 means no patching
    if args.patch_size == 0:
        args.patch_size = None
    train(args)


if __name__ == "__main__":
    main()
