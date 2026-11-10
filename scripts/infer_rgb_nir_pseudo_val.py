"""Infer and visualize the exact validation split from pseudo-disparity training."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from core.utils.utils import InputPadder  # noqa: E402
from Utils import AMP_DTYPE  # noqa: E402
from run_demo_rectified import load_calibration, rectify_pair, scaled_rectified_intrinsics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pseudo-disparity best_model.pt")
    parser.add_argument(
        "--val-manifest",
        type=Path,
        help="Validation val.txt (default: directory containing checkpoint/val.txt)",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=ROOT / "weights/23-36-37/model_best_bp2_serialize.pth",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=ROOT / "demo_data/rgb_nir_stereo_calibration.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output_pseudo_val")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--valid-iters", type=int, help="default: checkpoint training value")
    parser.add_argument("--max-disp", type=int, help="default: checkpoint training value")
    parser.add_argument("--alpha", type=float, help="default: checkpoint training value")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Path | str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {path}")
    samples = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) not in (5, 6):
                raise ValueError(
                    f"Expected 5 or 6 tab-separated fields in {path}:{line_number}, "
                    f"got {len(fields)}"
                )
            key, rgb, nir, disparity, mask = fields[:5]
            samples.append(
                {
                    "key": key,
                    "rgb": Path(rgb),
                    "nir": Path(nir),
                    "disparity": Path(disparity),
                    "mask": Path(mask),
                }
            )
    if not samples:
        raise RuntimeError(f"No validation samples in {path}")
    for sample in samples:
        for name in ("rgb", "nir", "disparity", "mask"):
            path_value = sample[name]
            if not path_value.is_file():
                raise FileNotFoundError(f"Missing {name} for {sample['key']}: {path_value}")
    return samples


def load_model_and_settings(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state dict: {args.checkpoint}")
    run_config = checkpoint.get("run_config", {})
    valid_iters = int(args.valid_iters if args.valid_iters is not None else run_config.get("valid_iters", 22))
    max_disp = int(args.max_disp if args.max_disp is not None else run_config.get("max_disp", 192))
    checkpoint_alpha = run_config.get("rectification_alpha")
    alpha = args.alpha if args.alpha is not None else checkpoint_alpha

    model = torch.load(args.base_model, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.to(device).eval()
    return model, valid_iters, max_disp, alpha


def infer_pair(model, left_raw, right_raw, calibration, device, valid_iters):
    left, right, rectification_valid = rectify_pair(left_raw, right_raw, calibration)
    height, width = left.shape[:2]
    image0 = torch.as_tensor(left, device=device).float()[None].permute(0, 3, 1, 2)
    image1 = torch.as_tensor(right, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(image0.shape, divis_by=32, force_square=False)
    image0, image1 = padder.pad(image0, image1)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        prediction = model(
            image0,
            image1,
            iters=valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    disparity = padder.unpad(prediction.float()).cpu().numpy().reshape(height, width)
    disparity = np.clip(disparity, 0.0, None).astype(np.float32)
    _, xx = np.indices(disparity.shape)
    valid = rectification_valid & np.isfinite(disparity) & (disparity > 0.0)
    valid &= (xx - disparity) >= 0.0
    return left, right, disparity, valid


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (18, 18, 18), -1)
    cv2.putText(
        canvas, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas


def colorize(values: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if high > low:
        normalized[valid] = np.clip(
            (values[valid] - low) / (high - low) * 255.0, 0.0, 255.0
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(image[finite], [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return np.clip((image.astype(np.float32) - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def make_preview(
    left: np.ndarray,
    right: np.ndarray,
    prediction: np.ndarray,
    prediction_valid: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # Evaluate every supervised pseudo-label pixel. A zero or geometrically
    # invalid prediction is still an error and must not disappear from metrics.
    eval_mask = target_valid & np.isfinite(prediction)
    error = np.zeros_like(target, dtype=np.float32)
    error[eval_mask] = np.abs(prediction[eval_mask] - target[eval_mask])

    values = np.concatenate((prediction[prediction_valid], target[target_valid]))
    if values.size:
        disp_low, disp_high = np.percentile(values, [1.0, 99.0])
        if disp_high <= disp_low:
            disp_high = disp_low + 1.0
    else:
        disp_low, disp_high = 0.0, 1.0
    error_high = float(np.percentile(error[eval_mask], 99.0)) if eval_mask.any() else 1.0
    error_high = max(error_high, 1.0)

    prediction_vis = colorize(prediction, prediction_valid, float(disp_low), float(disp_high))
    target_vis = colorize(target, target_valid, float(disp_low), float(disp_high))
    error_vis = colorize(error, eval_mask, 0.0, error_high)
    panels = [
        add_title(to_rgb_uint8(left), "RECTIFIED RGB"),
        add_title(to_rgb_uint8(right), "RECTIFIED NIR"),
        add_title(prediction_vis, "PREDICTED DISPARITY"),
        add_title(target_vis, "PSEUDO GT"),
        add_title(error_vis, "ABS ERROR"),
    ]
    metrics = disparity_metrics(prediction, target, eval_mask)
    return np.concatenate(panels, axis=1), error, metrics


def disparity_metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    count = int(valid.sum())
    if count == 0:
        return {"valid_pixels": 0, "abs_error_sum": 0.0, "bad1_count": 0, "bad3_count": 0}
    error = np.abs(prediction[valid].astype(np.float64) - target[valid].astype(np.float64))
    return {
        "valid_pixels": count,
        "abs_error_sum": float(error.sum()),
        "bad1_count": int((error > 1.0).sum()),
        "bad3_count": int((error > 3.0).sum()),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}; CUDA device count={torch.cuda.device_count()}")

    manifest = args.val_manifest or args.checkpoint.parent / "val.txt"
    samples = read_manifest(manifest)
    if args.limit is not None:
        samples = samples[: args.limit]

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model, valid_iters, max_disp, alpha = load_model_and_settings(args, device)
    calibration = load_calibration(args.calibration_file, alpha_override=alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        name: args.output_dir / name
        for name in ("preview", "disparity_npy", "depth_npy", "valid_mask", "error_npy")
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Validation samples=%d, valid_iters=%d, max_disp=%d, alpha=%.6g",
        len(samples), valid_iters, max_disp, calibration["alpha"],
    )
    totals = {"valid_pixels": 0, "abs_error_sum": 0.0, "bad1_count": 0, "bad3_count": 0}
    records = []
    for index, sample in enumerate(samples, 1):
        key = str(sample["key"])
        preview_path = dirs["preview"] / f"{key}_val_comparison.png"
        left_raw = imageio.imread(sample["rgb"])
        right_raw = imageio.imread(sample["nir"])
        target = np.load(sample["disparity"], allow_pickle=False).astype(np.float32)
        target_mask_image = cv2.imread(str(sample["mask"]), cv2.IMREAD_UNCHANGED)
        if target_mask_image is None:
            raise ValueError(f"Cannot read target mask: {sample['mask']}")
        if target.shape != target_mask_image.shape:
            raise ValueError(
                f"Target/mask shape mismatch for {key}: {target.shape} vs {target_mask_image.shape}"
            )
        target_valid = (
            (target_mask_image > 0)
            & np.isfinite(target)
            & (target > 0.0)
            & (target < max_disp)
        )

        left, right, prediction, prediction_valid = infer_pair(
            model, left_raw, right_raw, calibration, device, valid_iters
        )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for {key}: {prediction.shape} vs {target.shape}"
            )
        preview, error, metrics = make_preview(
            left, right, prediction, prediction_valid, target, target_valid
        )
        imageio.imwrite(preview_path, preview)
        np.save(dirs["disparity_npy"] / f"{key}_disp_px.npy", prediction, allow_pickle=False)
        np.save(dirs["error_npy"] / f"{key}_abs_error_px.npy", error, allow_pickle=False)
        imageio.imwrite(
            dirs["valid_mask"] / f"{key}_eval_mask.png",
            (target_valid & np.isfinite(prediction)).astype(np.uint8) * 255,
        )

        k_rectified = scaled_rectified_intrinsics(
            calibration["P1"], calibration["size"], prediction.shape[::-1]
        )
        depth = np.zeros_like(prediction, dtype=np.float32)
        depth[prediction_valid] = (
            float(k_rectified[0, 0]) * float(calibration["baseline_m"])
            / prediction[prediction_valid]
        )
        np.save(dirs["depth_npy"] / f"{key}_depth_m.npy", depth, allow_pickle=False)

        for name in totals:
            totals[name] += metrics[name]
        record = {
            "key": key,
            "valid_pixels": metrics["valid_pixels"],
            "epe_px": metrics["abs_error_sum"] / max(metrics["valid_pixels"], 1),
            "bad1": metrics["bad1_count"] / max(metrics["valid_pixels"], 1),
            "bad3": metrics["bad3_count"] / max(metrics["valid_pixels"], 1),
        }
        records.append(record)
        logging.info(
            "[%d/%d] %s EPE=%.4f Bad1=%.2f%% Bad3=%.2f%% valid=%d",
            index, len(samples), key, record["epe_px"],
            100.0 * record["bad1"], 100.0 * record["bad3"], record["valid_pixels"],
        )

    count = totals["valid_pixels"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "base_model": str(args.base_model),
        "val_manifest": str(manifest),
        "calibration_file": str(args.calibration_file),
        "rectification_alpha": calibration["alpha"],
        "valid_iters": valid_iters,
        "max_disp": max_disp,
        "requested_samples": len(samples),
        "processed_samples": len(records),
        "valid_pixels": count,
        "epe_px": totals["abs_error_sum"] / count if count else None,
        "bad1": totals["bad1_count"] / count if count else None,
        "bad3": totals["bad3_count"] / count if count else None,
        "frames": records,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info(
        "Complete: frames=%d EPE=%s Bad1=%s Bad3=%s output=%s",
        len(records),
        "n/a" if summary["epe_px"] is None else f"{summary['epe_px']:.4f}px",
        "n/a" if summary["bad1"] is None else f"{100.0 * summary['bad1']:.2f}%",
        "n/a" if summary["bad3"] is None else f"{100.0 * summary['bad3']:.2f}%",
        args.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
