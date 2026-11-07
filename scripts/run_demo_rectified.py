"""Fast-FoundationStereo inference for raw RGB-left/NIR-right images.

The raw images are undistorted and stereo-rectified before inference. The
rectified projection matrix is then used for disparity-to-depth conversion.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.append(str(ROOT))

from core.utils.utils import InputPadder
from Utils import (
    AMP_DTYPE,
    depth2xyzmap,
    o3d,
    set_logging_format,
    set_seed,
    toOpen3dCloud,
    vis_disparity,
)


def ensure_three_channels(image, label):
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[..., :3]
    raise ValueError(f"{label} must be HxW grayscale or HxWxC; got {image.shape}")


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    required = (
        "image_size",
        "K_left",
        "dist_left",
        "K_right",
        "dist_right",
        "R_left_to_right",
        "T_left_to_right_m",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Calibration file is missing: {', '.join(missing)}")

    width, height = map(int, data["image_size"])
    K_left = np.asarray(data["K_left"], dtype=np.float64).reshape(3, 3)
    K_right = np.asarray(data["K_right"], dtype=np.float64).reshape(3, 3)
    dist_left = np.asarray(data["dist_left"], dtype=np.float64).reshape(-1)
    dist_right = np.asarray(data["dist_right"], dtype=np.float64).reshape(-1)
    rotation = np.asarray(data["R_left_to_right"], dtype=np.float64).reshape(3, 3)
    # OpenCV 5.x requires an explicit 3x1 translation column here. Older
    # OpenCV releases also accepted a flat (3,) vector.
    translation = np.asarray(
        data["T_left_to_right_m"], dtype=np.float64
    ).reshape(3, 1)
    alpha = float(data.get("alpha", 0.9))

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left,
        dist_left,
        K_right,
        dist_right,
        (width, height),
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    map_left = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1, (width, height), cv2.CV_32FC1
    )
    map_right = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2, (width, height), cv2.CV_32FC1
    )
    baseline_m = float(abs(P2[0, 3] / P2[0, 0]))
    return {
        "raw": data,
        "size": (width, height),
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi_left": roi_left,
        "roi_right": roi_right,
        "map_left": map_left,
        "map_right": map_right,
        "baseline_m": baseline_m,
    }


def rectify_pair(left, right, calibration):
    expected_width, expected_height = calibration["size"]
    for label, image in (("left", left), ("right", right)):
        height, width = image.shape[:2]
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"{label} image is {width}x{height}, but calibration is for "
                f"{expected_width}x{expected_height}. Do not resize raw images before rectification."
            )

    left = ensure_three_channels(left, "left image")
    if right.ndim == 3:
        right = right[..., 0] if right.shape[2] == 1 else cv2.cvtColor(right[..., :3], cv2.COLOR_RGB2GRAY)

    left_rectified = cv2.remap(
        left,
        calibration["map_left"][0],
        calibration["map_left"][1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    right_rectified = cv2.remap(
        right,
        calibration["map_right"][0],
        calibration["map_right"][1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    source_valid = np.full((expected_height, expected_width), 255, dtype=np.uint8)
    left_valid = cv2.remap(
        source_valid,
        calibration["map_left"][0],
        calibration["map_left"][1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    right_valid = cv2.remap(
        source_valid,
        calibration["map_right"][0],
        calibration["map_right"][1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rectification_valid = (left_valid > 0) & (right_valid > 0)
    return (
        left_rectified,
        ensure_three_channels(right_rectified, "rectified right image"),
        rectification_valid,
    )


def make_epipolar_preview(left, right, spacing=32):
    preview = np.concatenate([left, right], axis=1).copy()
    for y in range(spacing, preview.shape[0], spacing):
        cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 255, 0), 1)
    return preview


def scaled_rectified_intrinsics(P1, raw_size, output_size):
    raw_width, raw_height = raw_size
    width, height = output_size
    K = P1[:, :3].copy()
    K[0, :] *= width / raw_width
    K[1, :] *= height / raw_height
    return K


def array_list(array):
    return np.asarray(array).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"))
    parser.add_argument("--left_file", default=str(ROOT / "demo_data/left.png"))
    parser.add_argument("--right_file", default=str(ROOT / "demo_data/right.png"))
    parser.add_argument(
        "--calibration_file",
        default=str(ROOT / "demo_data/rgb_nir_stereo_calibration.json"),
        help="Raw RGB/NIR stereo calibration containing K, distortion, R and T",
    )
    parser.add_argument("--out_dir", default=str(ROOT / "output_rectified"))
    parser.add_argument("--remove_invisible", default=1, type=int)
    parser.add_argument("--denoise_cloud", default=0, type=int)
    parser.add_argument("--denoise_nb_points", default=30, type=int)
    parser.add_argument("--denoise_radius", default=0.03, type=float)
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--hiera", default=0, type=int)
    parser.add_argument("--get_pc", default=1, type=int)
    parser.add_argument("--valid_iters", default=8, type=int)
    parser.add_argument("--max_disp", default=192, type=int)
    parser.add_argument("--zfar", default=100.0, type=float)
    parser.add_argument("--display", default=0, type=int)
    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError("--scale must be positive")

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.model_dir).parent / "cfg.yaml", "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")

    calibration = load_calibration(args.calibration_file)
    left_raw = imageio.imread(args.left_file)
    right_raw = imageio.imread(args.right_file)
    left, right, rectification_valid = rectify_pair(left_raw, right_raw, calibration)
    imageio.imwrite(output_dir / "left_rectified.png", left)
    imageio.imwrite(output_dir / "right_rectified.png", right)
    imageio.imwrite(output_dir / "rectified_epipolar.png", make_epipolar_preview(left, right))

    if float(args.scale) != 1.0:
        left = cv2.resize(left, dsize=None, fx=float(args.scale), fy=float(args.scale), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
        rectification_valid = cv2.resize(
            rectification_valid.astype(np.uint8),
            dsize=(left.shape[1], left.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    height, width = left.shape[:2]
    K_rectified = scaled_rectified_intrinsics(
        calibration["P1"], calibration["size"], (width, height)
    )
    baseline_m = calibration["baseline_m"]
    logging.info(
        "Rectification applied: output=%dx%d, fx=%.9f px, baseline=%.12f m",
        width,
        height,
        K_rectified[0, 0],
        baseline_m,
    )

    model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()

    left_original = left.copy()
    right_original = right.copy()
    image0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    image1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(image0.shape, divis_by=32, force_square=False)
    image0, image1 = padder.pad(image0, image1)

    logging.info("Start forward; the first run can be slow due to compilation")
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        if not args.hiera:
            disparity = model.forward(
                image0,
                image1,
                iters=args.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        else:
            disparity = model.run_hierachical(
                image0, image1, iters=args.valid_iters, test_mode=True, small_ratio=0.5
            )
    disparity = padder.unpad(disparity.float()).cpu().numpy().reshape(height, width)
    disparity = np.clip(disparity, 0, None).astype(np.float32)
    np.save(output_dir / "disparity_pixels.npy", disparity)

    yy, xx = np.indices(disparity.shape)
    valid = rectification_valid & np.isfinite(disparity) & (disparity > 0)
    if args.remove_invisible:
        valid &= (xx - disparity) >= 0
    imageio.imwrite(output_dir / "valid_mask.png", (valid.astype(np.uint8) * 255))

    if valid.any():
        vis_min, vis_max = np.percentile(disparity[valid], [2.0, 98.0])
        if vis_max <= vis_min:
            vis_max = vis_min + 1.0
    else:
        vis_min, vis_max = 0.0, 1.0
    disparity_for_vis = disparity.copy()
    disparity_for_vis[~valid] = np.inf
    disparity_vis = vis_disparity(
        disparity_for_vis,
        min_val=float(vis_min),
        max_val=float(vis_max),
        invalid_thres=np.inf,
        cmap=None,
        color_map=cv2.COLORMAP_TURBO,
    )
    imageio.imwrite(
        output_dir / "disp_vis.png",
        np.concatenate([left_original, right_original, disparity_vis], axis=1),
    )

    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = K_rectified[0, 0] * baseline_m / disparity[valid]
    np.save(output_dir / "depth_meter.npy", depth)

    if args.get_pc:
        xyz_map = depth2xyzmap(depth, K_rectified)
        cloud = toOpen3dCloud(xyz_map.reshape(-1, 3), left_original.reshape(-1, 3))
        points = np.asarray(cloud.points)
        keep = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] <= args.zfar)
        cloud = cloud.select_by_index(np.flatnonzero(keep).tolist())
        o3d.io.write_point_cloud(str(output_dir / "cloud.ply"), cloud)

        if args.denoise_cloud:
            cloud = cloud.voxel_down_sample(voxel_size=0.001)
            _, indices = cloud.remove_radius_outlier(
                nb_points=args.denoise_nb_points, radius=args.denoise_radius
            )
            cloud = cloud.select_by_index(indices)
            o3d.io.write_point_cloud(str(output_dir / "cloud_denoise.ply"), cloud)

    metadata = {
        "left_raw": str(args.left_file),
        "right_raw": str(args.right_file),
        "model": str(args.model_dir),
        "calibration_file": str(args.calibration_file),
        "rectification": "cv2.stereoRectify + cv2.initUndistortRectifyMap + cv2.remap",
        "raw_size_wh": list(calibration["size"]),
        "inference_size_wh": [width, height],
        "P1": array_list(calibration["P1"]),
        "P2": array_list(calibration["P2"]),
        "Q": array_list(calibration["Q"]),
        "K_rectified_scaled": array_list(K_rectified),
        "fx_rectified_scaled_px": float(K_rectified[0, 0]),
        "baseline_m": baseline_m,
        "fx_times_baseline_px_m": float(K_rectified[0, 0] * baseline_m),
        "rectification_alpha": float(calibration["raw"].get("alpha", 0.0)),
        "rectification_valid_fraction": float(rectification_valid.mean()),
        "disparity_visualization_percentiles_px": [float(vis_min), float(vis_max)],
        "disparity_reference": "rectified left RGB image; d = x_left - x_right",
        "depth_conversion": "depth_m = fx_rectified_scaled_px * baseline_m / disparity_pixels",
        "roi_left": list(calibration["roi_left"]),
        "roi_right": list(calibration["roi_right"]),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    logging.info("Outputs saved to %s", output_dir)
