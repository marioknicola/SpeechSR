import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

MODEL_NAME_MAP = {
    "UNet-v2": "U-Net",
    "ResidualIII": "Proposed",
    "Bicubic": "Input",
}

FILENAME_PATTERN = re.compile(r"^(?P<model>.+)_Subject(?P<subject>\d{4})_(?P<token>.+)\.(nii|nii\.gz)$")


def normalize01(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v, max_v = float(img.min()), float(img.max())
    if max_v <= min_v:
        return np.zeros_like(img, dtype=np.float32)
    return (img - min_v) / (max_v - min_v)


def load_2d(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[:, :, 0]
    raise ValueError(f"Unsupported shape {data.shape} in {path}")


def find_hr(hr_dir: Path, subject: str, token: str) -> Path | None:
    for candidate in [
        hr_dir / f"HR_kspace_Subject{subject}_{token}.nii",
        hr_dir / f"HR_kspace_Subject{subject}_{token}.nii.gz",
    ]:
        if candidate.exists():
            return candidate
    return None


def compute_metrics(pred: np.ndarray, hr: np.ndarray) -> dict[str, float]:
    pred = normalize01(pred)
    hr = normalize01(hr)
    if pred.shape != hr.shape:
        pred = resize(pred, hr.shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    mse = float(np.mean((pred - hr) ** 2))
    psnr = float(peak_signal_noise_ratio(hr, pred, data_range=1.0))
    ssim = float(structural_similarity(hr, pred, data_range=1.0, win_size=7))
    return {"PSNR": psnr, "SSIM": ssim, "MSE": mse}


def collect(pred_dir: Path, input_dir: Path, hr_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for source_dir in [pred_dir, input_dir]:
        files = sorted(source_dir.glob("*.nii")) + sorted(source_dir.glob("*.nii.gz"))
        for file_path in files:
            match = FILENAME_PATTERN.match(file_path.name)
            if not match:
                continue
            model = match.group("model")
            if source_dir == input_dir and model != "Bicubic":
                continue

            subject = match.group("subject")
            token = match.group("token")
            hr_file = find_hr(hr_dir, subject, token)
            if hr_file is None:
                continue

            values = compute_metrics(load_2d(file_path), load_2d(hr_file))
            records.append({"model": MODEL_NAME_MAP.get(model, model), "subject": subject, "token": token, **values})

    return pd.DataFrame(records)


def plot(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for metric in ["PSNR", "SSIM", "MSE"]:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="model", y=metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{metric.lower()}.png", dpi=300)
        plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["PSNR", "SSIM", "MSE"]):
        sns.boxplot(data=df, x="model", y=metric, ax=ax)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_combined.png", dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation box plots from NIfTI predictions")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory with model predictions")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with Bicubic input baseline")
    parser.add_argument("--hr-dir", type=Path, required=True, help="Directory with HR ground-truth NIfTI")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = collect(args.pred_dir, args.input_dir, args.hr_dir)
    if frame.empty:
        raise ValueError("No valid prediction/HR pairs found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "metrics_per_image.csv", index=False)
    plot(frame, args.output_dir)
