import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def variable_density_mask(nx: int, center_width: int = 18, edge_acceleration: int = 3) -> np.ndarray:
    mask = np.zeros(nx, dtype=np.float32)
    center_start = max(0, nx // 2 - center_width // 2)
    center_end = min(nx, center_start + center_width)
    mask[center_start:center_end] = 1.0

    left_indices = np.arange(0, center_start, edge_acceleration)
    right_indices = np.arange(center_end, nx, edge_acceleration)
    mask[left_indices] = 1.0
    mask[right_indices] = 1.0
    return mask


def undersample_kspace(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if kspace.ndim not in {3, 4}:
        raise ValueError(f"Expected k-space shape (Ny,Nx,Nc) or (Ny,Nx,Nc,Nt), got {kspace.shape}")

    undersampled = np.zeros_like(kspace)
    if kspace.ndim == 3:
        undersampled[:, mask > 0, :] = kspace[:, mask > 0, :]
    else:
        undersampled[:, mask > 0, :, :] = kspace[:, mask > 0, :, :]
    return undersampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply variable-density synthetic undersampling to k-space MAT files")
    parser.add_argument("--input-mat", type=Path, required=True, help="Input .mat containing key `kspace`")
    parser.add_argument("--output-mat", type=Path, required=True)
    parser.add_argument("--center-width", type=int, default=18)
    parser.add_argument("--edge-acceleration", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = loadmat(args.input_mat)
    if "kspace" not in data:
        raise KeyError(f"Input MAT missing `kspace` key: {args.input_mat}")

    kspace = data["kspace"]
    nx = kspace.shape[1]
    mask = variable_density_mask(nx=nx, center_width=args.center_width, edge_acceleration=args.edge_acceleration)
    kspace_us = undersample_kspace(kspace, mask)

    args.output_mat.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output_mat, {"kspace": kspace_us, "mask": mask})
    print(f"Saved undersampled k-space to {args.output_mat}")
