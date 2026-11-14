"""Evaluate four RGB/NIR FFS checkpoints on InfiniDepth pseudo disparity.

Metrics are pooled over the exact same LR-consistency-mask pixels for every
model: disparity MAE/RMSE in pixels and percentages with error < 1, 3, 5 px.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from infer_rgb_nir_pseudo_val import (  # noqa: E402
    add_title,
    colorize,
    infer_pair,
    read_manifest,
    to_rgb_uint8,
)
from run_demo_rectified import load_calibration, rectify_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net", nargs=2, action="append", required=True,
        metavar=("NAME", "CHECKPOINT"),
        help="Model name and best_model.pt; specify exactly four times",
    )
    parser.add_argument(
        "--val-manifest", type=Path, required=True,
        help="InfiniDepth pseudo-disparity validation val.txt",
    )
    parser.add_argument(
        "--base-model", type=Path,
        default=ROOT / "weights/23-36-37/model_best_bp2_serialize.pth",
    )
    parser.add_argument(
        "--calibration-file", type=Path,
        default=ROOT / "demo_data/rgb_nir_stereo_calibration.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "eval_four_models_infinidepth_disp",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--valid-iters", type=int, help="Override checkpoint settings")
    parser.add_argument("--max-disp", type=float, default=192.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-previews", action="store_true",
        help="Skip RGB/NIR/GT/four-model comparison images",
    )
    args = parser.parse_args()
    if len(args.net) != 4:
        parser.error("Specify exactly four --net NAME CHECKPOINT arguments")
    if len({name for name, _ in args.net}) != 4:
        parser.error("The four --net names must be unique")
    if args.max_disp <= 0.0:
        parser.error("--max-disp must be positive")
    if args.valid_iters is not None and args.valid_iters < 1:
        parser.error("--valid-iters must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def safe_name(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not result:
        raise ValueError(f"Invalid model name: {name!r}")
    return result


def inspect_checkpoint(path: Path, valid_iters_override: int | None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected checkpoint with a model state dict: {path}")
    state_dict = checkpoint["model"]
    if any(key.startswith("base_model.feature.fusion_") for key in state_dict):
        raise ValueError(
            f"This evaluator accepts RGB/NIR-only checkpoints, not PIM/DoLP: {path}"
        )
    run_config = checkpoint.get("run_config", {})
    valid_iters = int(
        valid_iters_override
        if valid_iters_override is not None
        else run_config.get("valid_iters", 22)
    )
    del checkpoint
    return {"valid_iters": valid_iters}


def load_model(
    checkpoint_path: Path,
    base_model_path: Path,
    valid_iters: int,
    max_disp: float,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = torch.load(base_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.args.valid_iters = valid_iters
    model.args.max_disp = int(max_disp)
    return model.to(device).eval()


def read_target(sample: dict, max_disp: float) -> tuple[np.ndarray, np.ndarray]:
    disparity = np.load(sample["disparity"], allow_pickle=False).astype(np.float32)
    mask_image = cv2.imread(str(sample["mask"]), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise ValueError(f"Cannot read InfiniDepth mask: {sample['mask']}")
    if disparity.shape != mask_image.shape:
        raise ValueError(
            f"Disparity/mask shape mismatch for {sample['key']}: "
            f"{disparity.shape} vs {mask_image.shape}"
        )
    valid = (
        (mask_image > 0)
        & np.isfinite(disparity)
        & (disparity > 0.0)
        & (disparity < max_disp)
    )
    return disparity, valid


def pixel_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_valid: np.ndarray,
    max_disp: float,
) -> dict:
    count = int(target_valid.sum())
    if count == 0:
        return {
            "valid_pixels": 0, "invalid_predictions": 0,
            "abs_sum": 0.0, "sq_sum": 0.0,
            "lt1_count": 0, "lt3_count": 0, "lt5_count": 0,
        }
    raw = prediction[target_valid].astype(np.float64)
    finite = np.isfinite(raw)
    # Invalid predictions remain in the denominator and receive a large error.
    evaluated = np.nan_to_num(
        raw, nan=0.0, posinf=float(max_disp), neginf=0.0
    )
    error = np.abs(evaluated - target[target_valid].astype(np.float64))
    return {
        "valid_pixels": count,
        "invalid_predictions": int((~finite).sum()),
        "abs_sum": float(error.sum()),
        "sq_sum": float(np.square(error).sum()),
        "lt1_count": int((error < 1.0).sum()),
        "lt3_count": int((error < 3.0).sum()),
        "lt5_count": int((error < 5.0).sum()),
    }


def summarize(totals: dict, *, require_nonempty: bool = True) -> dict:
    count = totals["valid_pixels"]
    if count == 0:
        if require_nonempty:
            raise RuntimeError("No valid InfiniDepth pseudo-disparity pixels were evaluated")
        return {
            "valid_pixels": 0,
            "invalid_predictions": totals["invalid_predictions"],
            "mae_px": None,
            "rmse_px": None,
            "error_lt_1px_ratio": None,
            "error_lt_3px_ratio": None,
            "error_lt_5px_ratio": None,
            "error_lt_1px_percent": None,
            "error_lt_3px_percent": None,
            "error_lt_5px_percent": None,
        }
    return {
        "valid_pixels": count,
        "invalid_predictions": totals["invalid_predictions"],
        "mae_px": totals["abs_sum"] / count,
        "rmse_px": (totals["sq_sum"] / count) ** 0.5,
        "error_lt_1px_ratio": totals["lt1_count"] / count,
        "error_lt_3px_ratio": totals["lt3_count"] / count,
        "error_lt_5px_ratio": totals["lt5_count"] / count,
        "error_lt_1px_percent": 100.0 * totals["lt1_count"] / count,
        "error_lt_3px_percent": 100.0 * totals["lt3_count"] / count,
        "error_lt_5px_percent": 100.0 * totals["lt5_count"] / count,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}; CUDA device count={torch.cuda.device_count()}")
    if not args.base_model.is_file():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    samples = read_manifest(args.val_manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    models = []
    slugs = set()
    for name, checkpoint_text in args.net:
        slug = safe_name(name)
        if slug in slugs:
            raise ValueError(f"Model names produce duplicate output names: {slug}")
        slugs.add(slug)
        path = Path(checkpoint_text)
        models.append({
            "name": name,
            "slug": slug,
            "checkpoint": path,
            **inspect_checkpoint(path, args.valid_iters),
        })

    calibration = load_calibration(args.calibration_file, alpha_override=args.alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_root = args.output_dir / "predictions"
    preview_dir = args.output_dir / "preview"
    predictions_root.mkdir(parents=True, exist_ok=True)
    if not args.no_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model_results = {}
    for model_index, spec in enumerate(models, 1):
        logging.info(
            "Loading model %d/4: %s; valid_iters=%d",
            model_index, spec["name"], spec["valid_iters"],
        )
        model = load_model(
            spec["checkpoint"], args.base_model, spec["valid_iters"],
            args.max_disp, device,
        )
        prediction_dir = predictions_root / spec["slug"]
        prediction_dir.mkdir(parents=True, exist_ok=True)
        totals = {
            "valid_pixels": 0, "invalid_predictions": 0,
            "abs_sum": 0.0, "sq_sum": 0.0,
            "lt1_count": 0, "lt3_count": 0, "lt5_count": 0,
        }
        frame_results = []
        for sample_index, sample in enumerate(samples, 1):
            left_raw = imageio.imread(sample["rgb"])
            right_raw = imageio.imread(sample["nir"])
            _, _, prediction, _ = infer_pair(
                model, left_raw, right_raw, calibration, device,
                spec["valid_iters"],
            )
            target, target_valid = read_target(sample, args.max_disp)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch for {spec['name']}/"
                    f"{sample['key']}: {prediction.shape} vs {target.shape}"
                )
            metrics = pixel_metrics(prediction, target, target_valid, args.max_disp)
            for key in totals:
                totals[key] += metrics[key]
            frame_results.append({
                "key": str(sample["key"]),
                **summarize(metrics, require_nonempty=False),
            })
            np.save(
                prediction_dir / f"{sample['key']}_disp_px.npy",
                prediction, allow_pickle=False,
            )
            if sample_index == 1 or sample_index == len(samples) or sample_index % 20 == 0:
                logging.info(
                    "%s: %d/%d", spec["name"], sample_index, len(samples)
                )
        model_results[spec["name"]] = {
            "checkpoint": str(spec["checkpoint"]),
            "valid_iters": spec["valid_iters"],
            **summarize(totals),
            "frames": frame_results,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if not args.no_previews:
        logging.info("Generating %d comparison previews", len(samples))
        for sample_index, sample in enumerate(samples, 1):
            left_raw = imageio.imread(sample["rgb"])
            right_raw = imageio.imread(sample["nir"])
            left, right, _ = rectify_pair(left_raw, right_raw, calibration)
            target, target_valid = read_target(sample, args.max_disp)
            predictions = []
            values = [target[target_valid]]
            for spec in models:
                prediction = np.load(
                    predictions_root / spec["slug"] / f"{sample['key']}_disp_px.npy",
                    allow_pickle=False,
                )
                prediction_valid = np.isfinite(prediction) & (prediction > 0.0)
                predictions.append((spec["name"], prediction, prediction_valid))
                values.append(prediction[prediction_valid])
            nonempty = [array for array in values if array.size]
            if nonempty:
                low, high = np.percentile(np.concatenate(nonempty), [1.0, 99.0])
                if high <= low:
                    high = low + 1.0
            else:
                low, high = 0.0, 1.0
            panels = [
                add_title(to_rgb_uint8(left), "RECTIFIED RGB"),
                add_title(to_rgb_uint8(right), "RECTIFIED NIR"),
                add_title(
                    colorize(target, target_valid, float(low), float(high)),
                    "INFINIDEPTH PSEUDO GT",
                ),
            ]
            for name, prediction, valid in predictions:
                panels.append(add_title(
                    colorize(prediction, valid, float(low), float(high)), name
                ))
            imageio.imwrite(
                preview_dir / f"{sample['key']}_comparison.png",
                np.concatenate(panels, axis=1),
            )
            if sample_index == 1 or sample_index == len(samples) or sample_index % 20 == 0:
                logging.info("preview: %d/%d", sample_index, len(samples))

    summary = {
        "evaluation_target": "InfiniDepth left pseudo disparity on LR-consistency mask",
        "aggregation": "global pixel pooling over identical target-valid pixels",
        "threshold_definition": "strict absolute disparity error < 1, 3, 5 pixels",
        "val_manifest": str(args.val_manifest),
        "calibration_file": str(args.calibration_file),
        "rectification_alpha": calibration["alpha"],
        "max_disp_px": args.max_disp,
        "samples": len(samples),
        "models": model_results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "model", "valid_pixels", "invalid_predictions", "mae_px", "rmse_px",
            "error_lt_1px_percent", "error_lt_3px_percent", "error_lt_5px_percent",
            "checkpoint",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in model_results.items():
            writer.writerow({key: result.get(key) for key in fieldnames} | {"model": name})

    with (args.output_dir / "per_frame_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "model", "key", "valid_pixels", "invalid_predictions",
            "mae_px", "rmse_px", "error_lt_1px_percent",
            "error_lt_3px_percent", "error_lt_5px_percent",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in model_results.items():
            for frame in result["frames"]:
                writer.writerow(
                    {key: frame.get(key) for key in fieldnames} | {"model": name}
                )

    for name, result in model_results.items():
        logging.info(
            "%s: MAE=%.4fpx RMSE=%.4fpx e<1=%.2f%% e<3=%.2f%% e<5=%.2f%%",
            name, result["mae_px"], result["rmse_px"],
            result["error_lt_1px_percent"], result["error_lt_3px_percent"],
            result["error_lt_5px_percent"],
        )
    logging.info("Complete: %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
