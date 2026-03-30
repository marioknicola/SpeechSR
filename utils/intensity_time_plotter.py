import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def normalize01(frame: np.ndarray) -> np.ndarray:
    frame = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v, max_v = float(frame.min()), float(frame.max())
    if max_v <= min_v:
        return np.zeros_like(frame, dtype=np.float32)
    return (frame - min_v) / (max_v - min_v)


def sample_line(frame: np.ndarray, p1: tuple[float, float], p2: tuple[float, float], n: int) -> np.ndarray:
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    cols = np.clip(np.round(xs).astype(int), 0, frame.shape[1] - 1)
    rows = np.clip(np.round(ys).astype(int), 0, frame.shape[0] - 1)
    return frame[rows, cols]


def build_mmode(volume_path: Path, p1: tuple[float, float], p2: tuple[float, float], output_dir: Path) -> None:
    volume = nib.load(str(volume_path)).get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected dynamic 3D volume (H,W,T). Got shape {volume.shape}")

    h, w, t = volume.shape
    n = max(64, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 2))

    mmode = np.zeros((n, t), dtype=np.float32)
    for frame_idx in range(t):
        frame = normalize01(volume[:, :, frame_idx])
        mmode[:, frame_idx] = sample_line(frame, p1, p2, n)

    mmode = normalize01(mmode)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_frame = normalize01(volume[:, :, 0])
    fig_ref, ax_ref = plt.subplots(figsize=(6, 6))
    ax_ref.imshow(ref_frame.T, cmap="gray", origin="lower")
    ax_ref.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=2)
    ax_ref.scatter([p1[0], p2[0]], [p1[1], p2[1]], c=["lime", "cyan"], edgecolors="white", s=40)
    ax_ref.set_title("Reference frame and profile line")
    fig_ref.tight_layout()
    fig_ref.savefig(output_dir / f"{volume_path.stem}_reference_line.png", dpi=300)
    plt.close(fig_ref)

    fig_m, ax_m = plt.subplots(figsize=(12, 4))
    ax_m.imshow(mmode, cmap="gray", aspect="auto", origin="lower")
    ax_m.set_xlabel("Time frame")
    ax_m.set_ylabel("Position along line")
    ax_m.set_title("Intensity-time plot (M-mode)")
    fig_m.tight_layout()
    fig_m.savefig(output_dir / f"{volume_path.stem}_mmode.png", dpi=300)
    plt.close(fig_m)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate intensity-time (M-mode) plots from dynamic MRI")
    parser.add_argument("--volume", type=Path, required=True, help="Path to dynamic NIfTI volume (H,W,T)")
    parser.add_argument("--line", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intensity_time"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x1, y1, x2, y2 = args.line
    build_mmode(args.volume, (x1, y1), (x2, y2), args.output_dir)
