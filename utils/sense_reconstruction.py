import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.fft
import scipy.linalg
from scipy.io import loadmat


def sense_reconstruct(kspace: np.ndarray, coilmap: np.ndarray, acquired_indices: np.ndarray) -> np.ndarray:
    """Generalized SENSE reconstruction for k-space of shape (Ny, Nx, Nc, Nt)."""
    if kspace.ndim == 3:
        kspace = kspace[:, :, :, None]
    if coilmap.ndim == 3:
        coilmap = coilmap[:, :, :, None]

    ny, nx, nc, nt = kspace.shape
    recon = np.zeros((ny, nx, nt), dtype=np.complex64)

    k_mask = np.zeros(nx, dtype=np.complex64)
    k_mask[acquired_indices] = 1.0
    psf = scipy.fft.ifftshift(scipy.fft.ifft(scipy.fft.ifftshift(k_mask)))
    a_matrix = scipy.linalg.circulant(psf)

    for t in range(nt):
        img_aliased = scipy.fft.ifftshift(
            scipy.fft.ifft2(scipy.fft.ifftshift(kspace[:, :, :, t], axes=(0, 1)), axes=(0, 1)),
            axes=(0, 1),
        )
        sens = coilmap[:, :, :, t % coilmap.shape[3]]

        for y in range(ny):
            i_vec = img_aliased[y, :, :].T.flatten()
            e = np.zeros((nc * nx, nx), dtype=np.complex64)
            for c in range(nc):
                e[c * nx:(c + 1) * nx, :] = a_matrix @ np.diag(sens[y, :, c])

            e_h = e.conj().T
            lhs = e_h @ e
            rhs = e_h @ i_vec
            reg = 1e-6 * np.trace(lhs) / nx
            recon[y, :, t] = scipy.linalg.solve(lhs + np.eye(nx) * reg, rhs, assume_a="her")

    return np.abs(recon)


def save_nifti(volume: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4)), str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SENSE reconstruction from undersampled k-space and coil maps")
    parser.add_argument("--kspace-mat", type=Path, required=True, help="MAT file containing `kspace`")
    parser.add_argument("--coilmap-mat", type=Path, required=True, help="MAT file containing `coilmap`")
    parser.add_argument("--output-nii", type=Path, required=True)
    parser.add_argument(
        "--acquired-indices",
        type=int,
        nargs="+",
        default=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81],
    ) # this matches the acquired undersampling pattern (ASSET 3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kspace_mat = loadmat(args.kspace_mat)
    coilmap_mat = loadmat(args.coilmap_mat)

    if "kspace" not in kspace_mat:
        raise KeyError("`kspace` key not found in k-space MAT")
    if "coilmap" not in coilmap_mat:
        raise KeyError("`coilmap` key not found in coil map MAT")

    reconstructed = sense_reconstruct(
        kspace=kspace_mat["kspace"],
        coilmap=np.transpose(coilmap_mat["coilmap"], (0, 1, 3, 2)),
        acquired_indices=np.array(args.acquired_indices, dtype=np.int32),
    )
    save_nifti(reconstructed, args.output_nii)
    print(f"Saved SENSE reconstruction to {args.output_nii}")
