import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn

from models.edsr import EDSR
from models.proposed1 import build_proposed1
from models.proposed2 import build_proposed2
from models.srcnn import SRCNN
from models.unet import build_unet
from models.vdsr import VDSR


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v, max_v = float(arr.min()), float(arr.max())
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def load_nifti_frames(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f"Only 2D/3D NIfTI is supported. Got shape {data.shape} for {path}")
    return data, img


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model(model_name: str, temporal: bool = False) -> nn.Module:
    in_channels = 3 if temporal else 1
    if model_name == "unet":
        return build_unet(base_filters=32, bilinear=True)
    if model_name == "proposed1":
        return build_proposed1(n_res_blocks=16, n_feats=64, reduction=16, res_scale=0.1)
    if model_name in ("proposed", "proposed2"):
        return build_proposed2(n_res_blocks=16, n_feats=64, reduction=16, res_scale=0.1, in_channels=in_channels)
    if model_name == "srcnn":
        return SRCNN()
    if model_name == "vdsr":
        return VDSR()
    if model_name == "edsr":
        return EDSR(n_resblocks=16, n_feats=64, scale=4)
    raise ValueError(f"Unsupported model: {model_name}")


def extract_state_dict(state: object) -> dict[str, torch.Tensor]:
    if isinstance(state, dict):
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            return state["model_state_dict"]
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if all(isinstance(k, str) for k in state.keys()):
            return state
    raise ValueError("Checkpoint format not recognized.")


def run_inference(model: nn.Module, volume: np.ndarray, device: torch.device) -> np.ndarray:
    """Single-frame inference: each frame processed independently."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(volume.shape[2]):
            frame = normalize01(volume[:, :, i])
            tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(tensor).detach().cpu().squeeze().numpy().astype(np.float32)
            outputs.append(pred)
    return np.stack(outputs, axis=2)


def run_inference_temporal(model: nn.Module, volume: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Temporal inference: (t-1, t, t+1) stacked as a 3-channel input.

    Boundary frames are padded by repeating the edge frame. Provides the model
    with adjacent-frame context for temporal consistency across dynamic volumes.
    """
    model.eval()
    n_frames = volume.shape[2]
    frames = [normalize01(volume[:, :, i]) for i in range(n_frames)]
    outputs = []
    with torch.no_grad():
        for i in range(n_frames):
            prev_frame = frames[max(i - 1, 0)]
            curr_frame = frames[i]
            next_frame = frames[min(i + 1, n_frames - 1)]
            tensor = torch.from_numpy(np.stack([prev_frame, curr_frame, next_frame], axis=0))
            tensor = tensor.unsqueeze(0).to(device)
            pred = model(tensor).detach().cpu().squeeze().numpy().astype(np.float32)
            outputs.append(pred)
    return np.stack(outputs, axis=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with SpeechSR pretrained models")
    parser.add_argument("--model", choices=["unet", "proposed1", "proposed2", "proposed", "srcnn", "vdsr", "edsr"],
                        required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input NIfTI file (.nii/.nii.gz)")
    parser.add_argument("--output", type=str, required=True, help="Output NIfTI file")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument(
        "--temporal-window",
        type=int,
        choices=[1, 3],
        default=3,
        help="Temporal context window. 3 (default) = stack (t-1, t, t+1) as 3-channel input. "
             "1 = single-frame per-frame inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temporal = args.temporal_window == 3

    if temporal and args.model not in ("proposed", "proposed2"):
        raise ValueError("--temporal-window 3 is only supported for --model proposed / proposed2")

    model = build_model(args.model, temporal=temporal)

    default_ckpt = Path(__file__).parent / "models" / "pretrained" / f"{args.model}.pth"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_ckpt
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = pick_device(args.device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(state)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    volume, ref_img = load_nifti_frames(Path(args.input))

    if temporal:
        pred = run_inference_temporal(model, volume, device)
    else:
        pred = run_inference(model, volume, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(pred, affine=ref_img.affine, header=ref_img.header)
    nib.save(out_img, str(out_path))
    print(f"Saved inference output: {out_path}")


if __name__ == "__main__":
    main()
