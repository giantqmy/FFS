"""Compare multiple intensity-only and DoLP-PIM FFS checkpoints on one validation split."""

from __future__ import annotations

import argparse
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

from core.pim_ffs import FastFoundationStereoPIM  # noqa: E402
from infer_rgb_nir_pseudo_val import (  # noqa: E402
    add_title,
    colorize,
    disparity_metrics,
    infer_pair,
    read_manifest,
    to_rgb_uint8,
)
from run_demo_rectified import (  # noqa: E402
    load_calibration,
    rectify_pair,
    scaled_rectified_intrinsics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net", nargs=2, action="append", required=True,
        metavar=("NAME", "CHECKPOINT"),
        help="Model display name and best_model.pt; repeat for each model",
    )
    parser.add_argument(
        "--val-manifest", type=Path,
        help="Validation val.txt; default: val.txt beside the first checkpoint",
    )
    parser.add_argument(
        "--base-model", type=Path,
        default=ROOT / "weights/23-36-37/model_best_bp2_serialize.pth",
    )
    parser.add_argument(
        "--calibration-file", type=Path,
        default=ROOT / "demo_data/rgb_nir_stereo_calibration.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output_multi_model_val")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--valid-iters", type=int, help="Override every checkpoint setting")
    parser.add_argument("--max-disp", type=int, help="Override every checkpoint setting")
    parser.add_argument("--alpha", type=float, help="Override rectification alpha")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    names = [name for name, _ in args.net]
    if len(set(names)) != len(names):
        parser.error("Every --net NAME must be unique")
    return args


def safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not value:
        raise ValueError(f"Model name has no filesystem-safe characters: {name!r}")
    return value


def inspect_checkpoint(path: Path, args: argparse.Namespace) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected training checkpoint containing a 'model' state dict: {path}")
    config = checkpoint.get("run_config", {})
    state_dict = checkpoint["model"]
    is_pim = any(key.startswith("base_model.feature.fusion_") for key in state_dict)
    info = {
        "is_pim": is_pim,
        "valid_iters": int(
            args.valid_iters if args.valid_iters is not None else config.get("valid_iters", 22)
        ),
        "max_disp": int(
            args.max_disp if args.max_disp is not None else config.get("max_disp", 192)
        ),
        "alpha": float(
            args.alpha if args.alpha is not None else config.get("rectification_alpha", 0.0)
        ),
    }
    del checkpoint
    return info


def load_model(info: dict, checkpoint_path: Path, base_model_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = torch.load(base_model_path, map_location="cpu", weights_only=False)
    if info["is_pim"]:
        model = FastFoundationStereoPIM(model)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.args.valid_iters = info["valid_iters"]
    model.args.max_disp = info["max_disp"]
    return model.to(device).eval()


def read_target(sample: dict, max_disp: int) -> tuple[np.ndarray, np.ndarray]:
    target = np.load(sample["disparity"], allow_pickle=False).astype(np.float32)
    mask_image = cv2.imread(str(sample["mask"]), cv2.IMREAD_UNCHANGED)
    if mask_image is None:
        raise ValueError(f"Cannot read pseudo-label mask: {sample['mask']}")
    if target.shape != mask_image.shape:
        raise ValueError(
            f"Pseudo disparity/mask shape mismatch for {sample['key']}: "
            f"{target.shape} vs {mask_image.shape}"
        )
    valid = (
        (mask_image > 0) & np.isfinite(target)
        & (target > 0.0) & (target < max_disp)
    )
    return target, valid


def metric_record(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    metrics = disparity_metrics(prediction, target, valid & np.isfinite(prediction))
    count = metrics["valid_pixels"]
    return {
        "valid_pixels": count,
        "epe_px": metrics["abs_error_sum"] / count if count else None,
        "bad1": metrics["bad1_count"] / count if count else None,
        "bad3": metrics["bad3_count"] / count if count else None,
        "_abs_error_sum": metrics["abs_error_sum"],
        "_bad1_count": metrics["bad1_count"],
        "_bad3_count": metrics["bad3_count"],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-model inference requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}; CUDA device count={torch.cuda.device_count()}")
    if not args.base_model.is_file():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    nets = []
    used_slugs = set()
    for name, checkpoint_text in args.net:
        slug = safe_name(name)
        if slug in used_slugs:
            raise ValueError(f"Model names map to the same output directory: {slug}")
        used_slugs.add(slug)
        nets.append({"name": name, "slug": slug, "path": Path(checkpoint_text)})

    manifest = args.val_manifest or nets[0]["path"].parent / "val.txt"
    samples = read_manifest(manifest)
    if args.limit is not None:
        samples = samples[:args.limit]
    has_dolp = "rgb_dolp" in samples[0]

    for net in nets:
        net.update(inspect_checkpoint(net["path"], args))
        if net["is_pim"] and not has_dolp:
            raise ValueError(
                f"PIM model {net['name']!r} requires an 8-field PIM validation manifest"
            )
    alphas = {round(net["alpha"], 12) for net in nets}
    if len(alphas) != 1:
        details = ", ".join(f"{net['name']}={net['alpha']}" for net in nets)
        raise ValueError(f"Models use different rectification alpha values: {details}; pass --alpha")
    comparison_max_disp = args.max_disp or max(net["max_disp"] for net in nets)
    calibration = load_calibration(args.calibration_file, alpha_override=nets[0]["alpha"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for net in nets:
        root = args.output_dir / "predictions" / net["slug"]
        net["output"] = root
        for child in ("disparity_npy", "depth_npy", "valid_mask"):
            (root / child).mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    summaries = {}
    for model_index, net in enumerate(nets, 1):
        logging.info(
            "Loading model %d/%d: %s (%s)", model_index, len(nets), net["name"],
            "PIM" if net["is_pim"] else "intensity-only",
        )
        model = load_model(net, net["path"], args.base_model, device)
        totals = {"valid_pixels": 0, "abs_error_sum": 0.0, "bad1_count": 0, "bad3_count": 0}
        frames = []
        for sample_index, sample in enumerate(samples, 1):
            key = str(sample["key"])
            left_raw = imageio.imread(sample["rgb"])
            right_raw = imageio.imread(sample["nir"])
            if net["is_pim"]:
                left_dolp = cv2.imread(str(sample["rgb_dolp"]), cv2.IMREAD_GRAYSCALE)
                right_dolp = cv2.imread(str(sample["nir_dolp"]), cv2.IMREAD_GRAYSCALE)
            else:
                left_dolp = right_dolp = None
            _, _, prediction, prediction_valid = infer_pair(
                model, left_raw, right_raw, calibration, device, net["valid_iters"],
                left_dolp, right_dolp,
            )
            target, target_valid = read_target(sample, comparison_max_disp)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch for {net['name']}/{key}: "
                    f"{prediction.shape} vs {target.shape}"
                )
            record = metric_record(prediction, target, target_valid)
            totals["valid_pixels"] += record["valid_pixels"]
            totals["abs_error_sum"] += record.pop("_abs_error_sum")
            totals["bad1_count"] += record.pop("_bad1_count")
            totals["bad3_count"] += record.pop("_bad3_count")
            frames.append({"key": key, **record})

            np.save(net["output"] / "disparity_npy" / f"{key}_disp_px.npy", prediction)
            imageio.imwrite(
                net["output"] / "valid_mask" / f"{key}_valid.png",
                prediction_valid.astype(np.uint8) * 255,
            )
            k_rect = scaled_rectified_intrinsics(
                calibration["P1"], calibration["size"], prediction.shape[::-1]
            )
            depth = np.zeros_like(prediction, dtype=np.float32)
            depth[prediction_valid] = (
                float(k_rect[0, 0]) * float(calibration["baseline_m"])
                / prediction[prediction_valid]
            )
            np.save(net["output"] / "depth_npy" / f"{key}_depth_m.npy", depth)
            if sample_index == 1 or sample_index == len(samples) or sample_index % 20 == 0:
                logging.info("%s: %d/%d", net["name"], sample_index, len(samples))

        count = totals["valid_pixels"]
        summaries[net["name"]] = {
            "checkpoint": str(net["path"]),
            "model_type": "PIM" if net["is_pim"] else "intensity-only",
            "valid_iters": net["valid_iters"],
            "max_disp": net["max_disp"],
            "valid_pixels": count,
            "epe_px": totals["abs_error_sum"] / count if count else None,
            "bad1": totals["bad1_count"] / count if count else None,
            "bad3": totals["bad3_count"] / count if count else None,
            "frames": frames,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    logging.info("Generating %d comparison previews", len(samples))
    for sample_index, sample in enumerate(samples, 1):
        key = str(sample["key"])
        left_raw = imageio.imread(sample["rgb"])
        right_raw = imageio.imread(sample["nir"])
        left, right, _ = rectify_pair(left_raw, right_raw, calibration)
        target, target_valid = read_target(sample, comparison_max_disp)
        predictions = []
        all_values = [target[target_valid]]
        for net in nets:
            prediction = np.load(
                net["output"] / "disparity_npy" / f"{key}_disp_px.npy",
                allow_pickle=False,
            )
            valid_image = cv2.imread(
                str(net["output"] / "valid_mask" / f"{key}_valid.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            valid = valid_image > 0
            predictions.append((net, prediction, valid))
            all_values.append(prediction[valid])
        nonempty = [values for values in all_values if values.size]
        if nonempty:
            low, high = np.percentile(np.concatenate(nonempty), [1.0, 99.0])
            if high <= low:
                high = low + 1.0
        else:
            low, high = 0.0, 1.0
        panels = [
            add_title(to_rgb_uint8(left), "RECTIFIED RGB"),
            add_title(to_rgb_uint8(right), "RECTIFIED NIR"),
            add_title(colorize(target, target_valid, float(low), float(high)), "PSEUDO GT"),
        ]
        for net, prediction, valid in predictions:
            panels.append(
                add_title(
                    colorize(prediction, valid, float(low), float(high)),
                    f"{net['name']} ({'PIM' if net['is_pim'] else 'INTENSITY'})",
                )
            )
        imageio.imwrite(
            preview_dir / f"{key}_multi_model_comparison.png",
            np.concatenate(panels, axis=1),
        )
        if sample_index == 1 or sample_index == len(samples) or sample_index % 20 == 0:
            logging.info("preview: %d/%d", sample_index, len(samples))

    output_summary = {
        "val_manifest": str(manifest),
        "base_model": str(args.base_model),
        "calibration_file": str(args.calibration_file),
        "rectification_alpha": calibration["alpha"],
        "samples": len(samples),
        "panel_order": ["RECTIFIED RGB", "RECTIFIED NIR", "PSEUDO GT"]
        + [net["name"] for net in nets],
        "models": summaries,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(output_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("Complete: %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
