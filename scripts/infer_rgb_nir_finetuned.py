"""Rectify an RGB/NIR pair and infer disparity with a fine-tuned FFS checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from run_demo_rectified import load_calibration, rectify_pair, scaled_rectified_intrinsics

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent

import sys

sys.path.insert(0, str(ROOT))

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_model.pt from fine-tuning")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=ROOT / "weights/23-36-37/model_best_bp2_serialize.pth",
        help="Serialized FFS model used to construct the architecture",
    )
    parser.add_argument("--left-file", type=Path, required=True, help="Raw RGB left image")
    parser.add_argument("--right-file", type=Path, required=True, help="Raw NIR right image")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=ROOT / "demo_data/rgb_nir_stereo_calibration.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output_finetuned")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--min-disp-vis", type=float, default=None)
    parser.add_argument("--max-disp-vis", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU index {args.gpu}; available CUDA devices: {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(args.calibration_file)
    left_raw = imageio.imread(args.left_file)
    right_raw = imageio.imread(args.right_file)
    left, right, rectification_valid = rectify_pair(left_raw, right_raw, calibration)
    imageio.imwrite(args.output_dir / "left_rectified.png", left)
    imageio.imwrite(args.output_dir / "right_rectified.png", right)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state dict: {args.checkpoint}")
    model = torch.load(args.base_model, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.to(device).eval()

    height, width = left.shape[:2]
    image0 = torch.as_tensor(left, device=device).float()[None].permute(0, 3, 1, 2)
    image1 = torch.as_tensor(right, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(image0.shape, divis_by=32, force_square=False)
    image0, image1 = padder.pad(image0, image1)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        prediction = model(
            image0,
            image1,
            iters=args.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    disparity = padder.unpad(prediction.float()).cpu().numpy().reshape(height, width)
    disparity = np.clip(disparity, 0, None).astype(np.float32)

    yy, xx = np.indices(disparity.shape)
    valid = rectification_valid & np.isfinite(disparity) & (disparity > 0)
    valid &= (xx - disparity) >= 0
    np.save(args.output_dir / "disparity_pixels.npy", disparity)
    imageio.imwrite(args.output_dir / "valid_mask.png", valid.astype(np.uint8) * 255)

    if valid.any():
        vis_min = args.min_disp_vis
        vis_max = args.max_disp_vis
        if vis_min is None or vis_max is None:
            pmin, pmax = np.percentile(disparity[valid], [2.0, 98.0])
            vis_min = pmin if vis_min is None else vis_min
            vis_max = pmax if vis_max is None else vis_max
        if vis_max <= vis_min:
            vis_max = vis_min + 1.0
    else:
        vis_min, vis_max = 0.0, 1.0

    disp_for_vis = disparity.copy()
    disp_for_vis[~valid] = np.inf
    disp_vis = vis_disparity(
        disp_for_vis,
        min_val=float(vis_min),
        max_val=float(vis_max),
        invalid_thres=np.inf,
        color_map=cv2.COLORMAP_TURBO,
    )
    imageio.imwrite(args.output_dir / "disparity_vis.png", disp_vis)
    imageio.imwrite(
        args.output_dir / "pair_and_disparity.png",
        np.concatenate([left, right, disp_vis], axis=1),
    )

    K_rectified = scaled_rectified_intrinsics(
        calibration["P1"], calibration["size"], (width, height)
    )
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = (
        K_rectified[0, 0] * calibration["baseline_m"] / disparity[valid]
    )
    np.save(args.output_dir / "depth_meter.npy", depth)
    logging.info(
        "Saved rectified disparity to %s (valid pixels: %d)",
        args.output_dir,
        int(valid.sum()),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
