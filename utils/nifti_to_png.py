import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def normalize01(frame: np.ndarray) -> np.ndarray:
    frame = np.nan_to_num(frame.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v, max_v = float(frame.min()), float(frame.max())
    if max_v <= min_v:
        return np.zeros_like(frame, dtype=np.float32)
    return (frame - min_v) / (max_v - min_v)


def save_frame_png(frame: np.ndarray, output_path: Path) -> None:
    img = (normalize01(frame) * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)


def convert_nifti_to_png(input_dir: Path, output_dir: Path, all_frames: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nifti_files = sorted(input_dir.glob("*.nii")) + sorted(input_dir.glob("*.nii.gz"))

    for file_path in nifti_files:
        volume = nib.load(str(file_path)).get_fdata(dtype=np.float32)
        if volume.ndim == 2:
            save_frame_png(volume, output_dir / f"{file_path.stem}.png")
            continue

        if volume.ndim != 3:
            print(f"Skipping {file_path.name}: unsupported shape {volume.shape}")
            continue

        if all_frames:
            for frame_idx in range(volume.shape[2]):
                save_frame_png(volume[:, :, frame_idx], output_dir / f"{file_path.stem}_f{frame_idx:03d}.png")
        else:
            center = volume.shape[2] // 2
            save_frame_png(volume[:, :, center], output_dir / f"{file_path.stem}_f{center:03d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NIfTI files to PNG images")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-frames", action="store_true", help="Save every frame for 3D volumes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_nifti_to_png(args.input_dir, args.output_dir, args.all_frames)
