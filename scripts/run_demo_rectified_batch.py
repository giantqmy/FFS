"""Batch RGB/NIR rectification and inference with an official FFS model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(ROOT))

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity
from run_demo_rectified import load_calibration, rectify_pair, scaled_rectified_intrinsics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "weights/23-36-37/model_best_bp2_serialize.pth",
        help="Official serialized FFS model file",
    )
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--nir-dir", type=Path, required=True)
    parser.add_argument("--rgb-suffix", default="_S0_rgb.png")
    parser.add_argument("--nir-suffix", default="_S0_nir.png")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=ROOT / "demo_data/rgb_nir_stereo_calibration.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output_official_batch")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--min-disp-vis", type=float, default=None)
    parser.add_argument("--max-disp-vis", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-invisible", action="store_true")
    return parser.parse_args()


def index_images(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {directory}")
    return {path.name[:-len(suffix)]: path for path in directory.glob(f"*{suffix}")}


def infer_pair(model, left_raw, right_raw, calibration, device, args):
    left, right, rectification_valid = rectify_pair(left_raw, right_raw, calibration)
    height, width = left.shape[:2]
    image0 = torch.as_tensor(left, device=device).float()[None].permute(0, 3, 1, 2)
    image1 = torch.as_tensor(right, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(image0.shape, divis_by=32, force_square=False)
    image0, image1 = padder.pad(image0, image1)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disparity = model.forward(
            image0,
            image1,
            iters=args.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    disparity = padder.unpad(disparity.float()).cpu().numpy().reshape(height, width)
    disparity = np.clip(disparity, 0, None).astype(np.float32)

    yy, xx = np.indices(disparity.shape)
    valid = rectification_valid & np.isfinite(disparity) & (disparity > 0)
    if not args.keep_invisible:
        valid &= (xx - disparity) >= 0
    K_rectified = scaled_rectified_intrinsics(
        calibration["P1"], calibration["size"], (width, height)
    )
    fx_rectified = float(K_rectified[0, 0])
    baseline_m = float(calibration["baseline_m"])
    depth_m = np.zeros_like(disparity, dtype=np.float32)
    depth_m[valid] = fx_rectified * baseline_m / disparity[valid]
    return left, right, disparity, valid, depth_m, fx_rectified, baseline_m


def save_pair(out_dir: Path, key: str, result, args) -> dict:
    left, right, disparity, valid, depth_m, fx_rectified, baseline_m = result
    folders = {
        name: out_dir / name
        for name in ("rectified", "disparity", "depth", "preview", "masks")
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(folders["rectified"] / f"{key}_rgb.png", left)
    imageio.imwrite(folders["rectified"] / f"{key}_nir.png", right)
    np.save(folders["disparity"] / f"{key}_disp_px.npy", disparity)
    imageio.imwrite(folders["masks"] / f"{key}_valid.png", valid.astype(np.uint8) * 255)

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
    disp_vis_input = disparity.copy()
    disp_vis_input[~valid] = np.inf
    disp_vis = vis_disparity(
        disp_vis_input,
        min_val=float(vis_min),
        max_val=float(vis_max),
        invalid_thres=np.inf,
        color_map=cv2.COLORMAP_TURBO,
    )
    imageio.imwrite(folders["disparity"] / f"{key}_disp_vis.png", disp_vis)
    imageio.imwrite(
        folders["preview"] / f"{key}_rgb_nir_disp.png",
        np.concatenate([left, right, disp_vis], axis=1),
    )

    np.save(folders["depth"] / f"{key}_depth_m.npy", depth_m)
    depth_u16 = np.clip(depth_m * 256.0, 0, 65535).astype(np.uint16)
    imageio.imwrite(folders["depth"] / f"{key}_depth_m_x256_u16.png", depth_u16)
    return {
        "key": key,
        "valid_pixels": int(valid.sum()),
        "fx_rectified_px": fx_rectified,
        "baseline_m": baseline_m,
        "fx_times_baseline_px_m": fx_rectified * baseline_m,
        "depth_formula": "depth_m = fx_rectified_px * baseline_m / disparity_pixels",
        "depth_png_encoding": "uint16 = round(depth_m * 256); zero invalid; saturates at 255.996 m",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU index {args.gpu}; available CUDA devices: {torch.cuda.device_count()}"
        )
    if (args.min_disp_vis is None) != (args.max_disp_vis is None):
        raise ValueError("Set both --min-disp-vis and --max-disp-vis, or neither")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(args.calibration_file)
    rgb_files = index_images(args.rgb_dir, args.rgb_suffix)
    nir_files = index_images(args.nir_dir, args.nir_suffix)
    keys = sorted(set(rgb_files) & set(nir_files))
    if args.limit is not None:
        keys = keys[:max(0, args.limit)]
    if not keys:
        raise RuntimeError(
            f"No paired inputs; RGB={len(rgb_files)}, NIR={len(nir_files)}, "
            f"expected suffixes {args.rgb_suffix!r} and {args.nir_suffix!r}"
        )
    logging.info("Matched %d RGB/NIR pairs", len(keys))

    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.to(device).eval()

    records = []
    for index, key in enumerate(keys, start=1):
        result = infer_pair(
            model,
            imageio.imread(rgb_files[key]),
            imageio.imread(nir_files[key]),
            calibration,
            device,
            args,
        )
        record = save_pair(args.out_dir, key, result, args)
        record["rgb_file"] = str(rgb_files[key])
        record["nir_file"] = str(nir_files[key])
        records.append(record)
        logging.info(
            "[%d/%d] %s valid=%d fx=%.3f px baseline=%.6f m",
            index, len(keys), key, record["valid_pixels"],
            record["fx_rectified_px"], record["baseline_m"],
        )

    metadata = {
        "model": str(args.model_dir),
        "calibration_file": str(args.calibration_file),
        "valid_iters": args.valid_iters,
        "max_disp": args.max_disp,
        "depth_coordinate_frame": "rectified left-camera Z depth",
        "pairs": records,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("Finished %d pairs; outputs: %s", len(records), args.out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
