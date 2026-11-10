"""Full fine-tuning of FFS with RGB/NIR + left/right DoLP wavelet-PIM fusion.

Supervision remains the LR-consistent rectified pseudo disparity used by
finetune_rgb_nir_pseudo_disp.py. Sparse LiDAR is used only for per-epoch
validation depth metrics and never contributes to the loss.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from core.pim_ffs import FastFoundationStereoPIM  # noqa: E402
from finetune_rgb_nir_depth import seed_everything, sequence_loss, to_yamlable  # noqa: E402
from finetune_rgb_nir_pseudo_disp import (  # noqa: E402
    RGBNIRPseudoDisparityDataset,
    disparity_metrics,
    load_calibration,
    read_pseudo_metadata,
    sparse_depth_metrics,
    verify_pseudo_geometry,
)


DEFAULT_DATA_ROOT = Path("/media/ddc/新加卷/hys/qmy/depth/data")
DEFAULT_MODEL = ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_CALIBRATION = ROOT / "demo_data" / "rgb_nir_stereo_calibration.json"
DEFAULT_OUTPUT = ROOT / "finetune_rgb_nir_dolp_pim_pseudo_disp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_DATA_ROOT / "RGBS0")
    parser.add_argument("--nir-dir", type=Path, default=DEFAULT_DATA_ROOT / "NIRS0")
    parser.add_argument("--rgb-dolp-dir", type=Path, default=DEFAULT_DATA_ROOT / "RGBDOLP_color")
    parser.add_argument(
        "--nir-dolp-dir", type=Path, default=DEFAULT_DATA_ROOT / "NIRuseI4" / "output_dolp"
    )
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument(
        "--sparse-gt-dir",
        type=Path,
        default=Path("/media/ddc/新加卷/hys/qmy/depth/bevneed/raw_left_sparse_depth_u16"),
    )
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--valid-iters", type=int, default=22)
    parser.add_argument("--max-disp", type=float, default=192.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-min-depth", type=float, default=0.1)
    parser.add_argument("--eval-max-depth", type=float, default=200.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0.0:
        parser.error("epochs and batch-size must be >= 1; lr must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0, 1]")
    if args.max_disp <= 0.0:
        parser.error("--max-disp must be positive")
    if args.eval_min_depth <= 0.0 or args.eval_max_depth <= args.eval_min_depth:
        parser.error("Require 0 < --eval-min-depth < --eval-max-depth")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    return args


class RGBNIRDoLPPseudoDataset(RGBNIRPseudoDisparityDataset):
    def __init__(self, *args, rgb_dolp_dir: Path, nir_dolp_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        rgb_suffix = "_dolp_color_rgb.png"
        nir_suffix = "_dolp_nir.png"
        rgb_dolp = {
            path.name[: -len(rgb_suffix)]: path
            for path in rgb_dolp_dir.glob(f"*{rgb_suffix}")
        }
        nir_dolp = {
            path.name[: -len(nir_suffix)]: path
            for path in nir_dolp_dir.glob(f"*{nir_suffix}")
        }
        missing = []
        self.dolp_paths = []
        for sample in self.samples:
            key = sample[-1]
            if key not in rgb_dolp:
                missing.append(f"RGB DoLP: {key}")
            if key not in nir_dolp:
                missing.append(f"NIR DoLP: {key}")
            if key in rgb_dolp and key in nir_dolp:
                self.dolp_paths.append((rgb_dolp[key], nir_dolp[key]))
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} DoLP files; examples: {missing[:5]}"
            )
        print(
            f"DoLP counts: RGB={len(rgb_dolp)}, NIR={len(nir_dolp)}, "
            f"matched={len(self.dolp_paths)}",
            flush=True,
        )

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        expected_w, expected_h = self.calibration["size"]
        rgb_path, nir_path = self.dolp_paths[index]
        rgb_dolp = imageio.imread(rgb_path)
        nir_dolp = imageio.imread(nir_path)
        if rgb_dolp.dtype != np.uint8 or rgb_dolp.shape != (expected_h, expected_w, 3):
            raise ValueError(
                f"RGB DoLP must be uint8 {expected_w}x{expected_h}x3: "
                f"{rgb_path}; got {rgb_dolp.dtype} {rgb_dolp.shape}"
            )
        if nir_dolp.ndim == 3 and nir_dolp.shape[2] == 1:
            nir_dolp = nir_dolp[..., 0]
        if nir_dolp.dtype != np.uint8 or nir_dolp.shape != (expected_h, expected_w):
            raise ValueError(
                f"NIR DoLP must be uint8 {expected_w}x{expected_h}: "
                f"{nir_path}; got {nir_dolp.dtype} {nir_dolp.shape}"
            )
        rgb_dolp = cv2.remap(
            rgb_dolp, *self.calibration["map_left"], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        nir_dolp = cv2.remap(
            nir_dolp, *self.calibration["map_right"], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        nir_dolp = np.repeat(nir_dolp[..., None], 3, axis=2)
        sample["left_dolp"] = torch.from_numpy(
            rgb_dolp.transpose(2, 0, 1).copy()
        ).float()
        sample["right_dolp"] = torch.from_numpy(
            nir_dolp.transpose(2, 0, 1).copy()
        ).float()
        return sample


def save_manifest(
    path: Path, dataset: RGBNIRDoLPPseudoDataset, indices: list[int]
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# key\trgb\tnir\trgb_dolp\tnir_dolp\tdisparity\tmask\tsparse_gt_depth\n"
        )
        for index in indices:
            rgb, nir, disparity, mask, sparse_gt, key = dataset.samples[index]
            rgb_dolp, nir_dolp = dataset.dolp_paths[index]
            paths = (rgb, nir, rgb_dolp, nir_dolp, disparity, mask, sparse_gt)
            stream.write(f"{key}\t" + "\t".join(str(path.resolve()) for path in paths) + "\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    metadata = read_pseudo_metadata(args.pseudo_root)
    metadata_alpha = float(metadata["alpha"])
    alpha = metadata_alpha if args.alpha is None else float(args.alpha)
    if not np.isclose(alpha, metadata_alpha, atol=1e-12):
        raise ValueError(
            f"--alpha={alpha} conflicts with pseudo metadata alpha={metadata_alpha}"
        )
    calibration = load_calibration(args.calibration_file, alpha)
    verify_pseudo_geometry(metadata, calibration)
    dataset = RGBNIRDoLPPseudoDataset(
        args.rgb_dir,
        args.nir_dir,
        args.pseudo_root,
        args.sparse_gt_dir,
        calibration,
        args.max_disp,
        rgb_dolp_dir=args.rgb_dolp_dir,
        nir_dolp_dir=args.nir_dolp_dir,
    )

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit is not None:
        indices = indices[: min(args.limit, len(indices))]
    if len(indices) < 2:
        raise RuntimeError("At least two matched samples are required")
    val_count = min(len(indices) - 1, max(1, int(len(indices) * args.val_fraction)))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    dataset.set_sparse_eval_indices(val_indices)
    args.output.mkdir(parents=True, exist_ok=True)
    save_manifest(args.output / "train.txt", dataset, train_indices)
    save_manifest(args.output / "val.txt", dataset, val_indices)

    train_sample = dataset[train_indices[0]]
    val_sample = dataset[val_indices[0]]
    valid_count = int(train_sample["valid"].sum())
    sparse_count = int(
        (
            (val_sample["sparse_gt_depth"] >= args.eval_min_depth)
            & (val_sample["sparse_gt_depth"] <= args.eval_max_depth)
        ).sum()
    )
    if valid_count == 0 or sparse_count == 0:
        raise RuntimeError(
            f"Data check failed: train pseudo valid={valid_count}, "
            f"validation sparse GT valid={sparse_count}"
        )
    print(
        f"train={len(train_indices)}, val={len(val_indices)}, "
        f"RGB={tuple(train_sample['left'].shape)}, NIR={tuple(train_sample['right'].shape)}, "
        f"RGB_DoLP={tuple(train_sample['left_dolp'].shape)}, "
        f"NIR_DoLP={tuple(train_sample['right_dolp'].shape)}, "
        f"pseudo_valid={valid_count}, sparse_eval_valid={sparse_count}, alpha={alpha}",
        flush=True,
    )
    if args.check_data_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("FFS PIM full fine-tuning requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}; CUDA device count={torch.cuda.device_count()}")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    base_model = torch.load(args.model, map_location="cpu", weights_only=False)
    model = FastFoundationStereoPIM(base_model)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = int(args.max_disp)
    model.to(device).train()
    # Full fine-tuning: base FFS, both copied DoLP encoders, and both PIMs.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(model.args.mixed_precision))

    start_epoch = 0
    best_val = float("inf")
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_val = float(checkpoint.get("best_val_depth_mae_m", best_val))
        history = list(checkpoint.get("history", []))

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    run_config = {
        **vars(args),
        "rgb_dir": str(args.rgb_dir),
        "nir_dir": str(args.nir_dir),
        "rgb_dolp_dir": str(args.rgb_dolp_dir),
        "nir_dolp_dir": str(args.nir_dolp_dir),
        "pseudo_root": str(args.pseudo_root),
        "sparse_gt_dir": str(args.sparse_gt_dir),
        "calibration_file": str(args.calibration_file),
        "model": str(args.model),
        "resume": None if args.resume is None else str(args.resume),
        "output": str(args.output),
        "architecture": "FFS_shared_S0_encoder_plus_independent_DoLP_encoders_wavelet_PIM_x4",
        "fusion_source": "GatedRCCBStereo/networks/pim_fusion.py",
        "fusion": "left RGB+RGB-DoLP; right NIR+NIR-DoLP",
        "fine_tuning": "full model",
        "trainable_parameters": trainable,
        "total_parameters": total,
        "rectification_alpha": alpha,
        "fx_rectified": calibration["fx"],
        "baseline_m": calibration["baseline"],
        "loss": "unchanged: initial SmoothL1 plus gamma-weighted iterative L1",
        "best_model_selection": "minimum validation sparse-LiDAR depth MAE",
    }
    (args.output / "run_config.json").write_text(
        json.dumps(to_yamlable(run_config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "cfg.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(to_yamlable(dict(model.args)), stream, sort_keys=False)
    print(f"Full fine-tuning parameters: {trainable:,}/{total:,}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_result = {"epoch": epoch + 1}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            loss_sum = 0.0
            batches = 0
            disparity_totals = {"count": 0, "abs_sum": 0.0, "bad1": 0, "bad3": 0}
            depth_totals = {
                "count": 0, "abs_sum": 0.0, "sq_sum": 0.0,
                "delta1": 0, "delta2": 0, "delta3": 0,
            }
            context = torch.enable_grad() if phase == "train" else torch.no_grad()
            with context:
                for step, batch in enumerate(loader, 1):
                    left = batch["left"].to(device, non_blocking=True)
                    right = batch["right"].to(device, non_blocking=True)
                    left_dolp = batch["left_dolp"].to(device, non_blocking=True)
                    right_dolp = batch["right_dolp"].to(device, non_blocking=True)
                    gt_disp = batch["disp"].to(device, non_blocking=True)
                    valid = batch["valid"].to(device, non_blocking=True).bool()
                    sparse_gt = batch["sparse_gt_depth"].to(device, non_blocking=True)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    init_disp, predictions = model(
                        left, right, left_dolp, right_dolp,
                        iters=args.valid_iters, test_mode=False,
                        optimize_build_volume="pytorch1",
                    )
                    loss, details = sequence_loss(
                        init_disp, predictions, gt_disp, valid, args.gamma
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss: {details}; keys={batch['key']}"
                        )
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        metrics = disparity_metrics(predictions[-1], gt_disp, valid)
                        for name in disparity_totals:
                            disparity_totals[name] += metrics[name]
                    else:
                        metrics = sparse_depth_metrics(
                            predictions[-1], sparse_gt,
                            calibration["fx"] * calibration["baseline"],
                            args.eval_min_depth, args.eval_max_depth,
                        )
                        for name in depth_totals:
                            depth_totals[name] += metrics[name]
                    loss_sum += float(loss.detach())
                    batches += 1
                    if step == 1 or step == len(loader) or step % 20 == 0:
                        print(
                            f"epoch {epoch + 1}/{args.epochs} {phase} {step}/{len(loader)} "
                            f"loss={details['loss']:.6f} "
                            f"init={details['loss_init_disp']:.6f} "
                            f"iter={details['loss_disp']:.6f}",
                            flush=True,
                        )

            epoch_result[f"{phase}_loss"] = loss_sum / max(batches, 1)
            if phase == "train":
                count = disparity_totals["count"]
                epoch_result["train_disp_mae_px"] = (
                    disparity_totals["abs_sum"] / count if count else None
                )
                epoch_result["train_valid_pixels"] = count
            else:
                count = depth_totals["count"]
                epoch_result["val_depth_mae_m"] = depth_totals["abs_sum"] / count if count else None
                epoch_result["val_depth_rmse_m"] = (
                    (depth_totals["sq_sum"] / count) ** 0.5 if count else None
                )
                for delta in ("delta1", "delta2", "delta3"):
                    epoch_result[f"val_{delta}"] = depth_totals[delta] / count if count else None
                epoch_result["val_sparse_gt_pixels"] = count

        selection_metric = epoch_result["val_depth_mae_m"]
        if selection_metric is None:
            raise RuntimeError("Validation produced no valid sparse-LiDAR pixels")
        history.append(epoch_result)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_val_depth_mae_m": min(best_val, selection_metric),
            "history": history,
            "run_config": run_config,
        }
        if selection_metric < best_val:
            best_val = selection_metric
            checkpoint["best_val_depth_mae_m"] = best_val
            torch.save(checkpoint, args.output / "best_model.pt")
            torch.save(model, args.output / "model_best_serialize.pth")
        torch.save(checkpoint, args.output / "latest.pt")
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
