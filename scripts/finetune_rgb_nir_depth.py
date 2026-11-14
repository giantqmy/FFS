"""Fine-tune Fast-FoundationStereo on rectified RGB-left/NIR-right pairs.

Dense depth PNGs are uint16 values in metres * 256. They are rectified into
the left-camera grid and converted to disparity using d = fx * baseline / z.
Zero depth remains invalid and never contributes to the loss.
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))


DATA_ROOT = Path(r"I:\KEDADATA\data\fog")
DEFAULT_RGB = DATA_ROOT / "RGBS0"
DEFAULT_NIR = DATA_ROOT / "NIRS0"
DEFAULT_DEPTH = DATA_ROOT / "depth"
DEFAULT_MODEL = ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_CALIBRATION = ROOT / "demo_data" / "rgb_nir_stereo_calibration.json"
DEFAULT_OUTPUT = ROOT / "finetune_rgb_nir_depth"
DEPTH_METRES_PER_UINT16 = 1.0 / 256.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_calibration(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = map(int, data["image_size"])
    k_left = np.asarray(data["K_left"], dtype=np.float64).reshape(3, 3)
    k_right = np.asarray(data["K_right"], dtype=np.float64).reshape(3, 3)
    d_left = np.asarray(data["dist_left"], dtype=np.float64).reshape(-1)
    d_right = np.asarray(data["dist_right"], dtype=np.float64).reshape(-1)
    rotation = np.asarray(data["R_left_to_right"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(data["T_left_to_right_m"], dtype=np.float64).reshape(3, 1)
    alpha = float(data.get("alpha", 0.0))
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k_left, d_left, k_right, d_right, (width, height), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )
    map_left = cv2.initUndistortRectifyMap(
        k_left, d_left, r1, p1, (width, height), cv2.CV_32FC1
    )
    map_right = cv2.initUndistortRectifyMap(
        k_right, d_right, r2, p2, (width, height), cv2.CV_32FC1
    )
    baseline = float(abs(p2[0, 3] / p2[0, 0]))
    return {
        "size": (width, height), "map_left": map_left, "map_right": map_right,
        "P1": p1, "P2": p2, "fx": float(p1[0, 0]), "baseline": baseline,
        "alpha": alpha,
    }


def index_files(directory: Path, suffix: str) -> dict[str, Path]:
    return {p.name.removesuffix(suffix): p for p in directory.glob(f"*{suffix}")}


def index_sparse_depth_files(directory: Path) -> dict[str, Path]:
    suffixes = (
        "_depth_gt_sparse_u16.png",
        "_depth_sparse_u16.png",
        "_depth_raw_u16.png",
        "_depth_gt_u16.png",
        "_depth_dense_u16.png",
    )
    indexed = {}
    for suffix in suffixes:
        indexed.update(index_files(directory, suffix))
    return indexed


class RGBNIRDenseDepthDataset(Dataset):
    def __init__(
        self,
        rgb_dir: Path,
        nir_dir: Path,
        depth_dir: Path,
        calibration: dict,
        max_disp: float,
        max_depth: float,
        sparse_depth: bool = False,
        depth_suffix: str | None = None,
    ) -> None:
        self.calibration = calibration
        self.max_disp = float(max_disp)
        self.max_depth = float(max_depth)
        rgb = index_files(rgb_dir, "_S0_rgb.png")
        nir = index_files(nir_dir, "_S0_nir.png")
        if depth_suffix is not None:
            depth = index_files(depth_dir, depth_suffix)
        elif sparse_depth:
            depth = index_sparse_depth_files(depth_dir)
        else:
            depth = index_files(depth_dir, "_depth_dense_u16.png")
        keys = sorted(set(rgb) & set(nir) & set(depth))
        if not keys:
            label = "sparse-depth" if sparse_depth else "dense-depth"
            raise RuntimeError(f"No matching RGB/NIR/{label} timestamp prefixes found")
        self.samples = [(rgb[k], nir[k], depth[k], k) for k in keys]
        counts = {"RGB": len(rgb), "NIR": len(nir), "depth": len(depth), "matched": len(keys)}
        print(f"Dataset counts: {counts}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        rgb_path, nir_path, depth_path, key = self.samples[index]
        rgb = imageio.imread(rgb_path)
        nir = imageio.imread(nir_path)
        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        expected_w, expected_h = self.calibration["size"]
        if rgb.shape[:2] != (expected_h, expected_w):
            raise ValueError(f"Wrong RGB size for {rgb_path}: {rgb.shape}")
        if nir.shape[:2] != (expected_h, expected_w):
            raise ValueError(f"Wrong NIR size for {nir_path}: {nir.shape}")
        if depth_u16 is None or depth_u16.dtype != np.uint16 or depth_u16.shape != (expected_h, expected_w):
            got = None if depth_u16 is None else (depth_u16.dtype, depth_u16.shape)
            raise ValueError(f"Depth must be uint16 {expected_w}x{expected_h}: {depth_path}; got {got}")

        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        elif rgb.ndim == 3:
            rgb = rgb[..., :3]
        else:
            raise ValueError(f"Unsupported RGB shape: {rgb.shape}")
        if nir.ndim == 3:
            nir = nir[..., 0] if nir.shape[2] == 1 else cv2.cvtColor(nir[..., :3], cv2.COLOR_RGB2GRAY)

        rgb = cv2.remap(rgb, *self.calibration["map_left"], cv2.INTER_LINEAR)
        nir = cv2.remap(nir, *self.calibration["map_right"], cv2.INTER_LINEAR)
        depth_m = depth_u16.astype(np.float32) * DEPTH_METRES_PER_UINT16
        depth_m = cv2.remap(depth_m, *self.calibration["map_left"], cv2.INTER_NEAREST)
        nir = np.repeat(nir[..., None], 3, axis=2)

        eval_valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= self.max_depth)
        eval_depth_m = np.where(eval_valid, depth_m, 0.0).astype(np.float32)
        valid = eval_valid.copy()
        disparity = np.zeros_like(depth_m, dtype=np.float32)
        disparity[valid] = (
            self.calibration["fx"] * self.calibration["baseline"] / depth_m[valid]
        )
        xx = np.arange(expected_w, dtype=np.float32)[None, :]
        valid &= (disparity > 0.0) & (disparity < self.max_disp) & ((xx - disparity) >= 0.0)
        depth_m[~valid] = 0.0
        disparity[~valid] = 0.0

        return {
            "left": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
            "right": torch.from_numpy(nir.transpose(2, 0, 1).copy()).float(),
            "depth": torch.from_numpy(depth_m[None].copy()).float(),
            "eval_depth": torch.from_numpy(eval_depth_m[None].copy()).float(),
            "eval_valid": torch.from_numpy(eval_valid[None].copy()),
            "disp": torch.from_numpy(disparity[None].copy()).float(),
            "valid": torch.from_numpy(valid[None].copy()),
            "key": key,
        }


def sequence_loss(
    init_disp: torch.Tensor,
    predictions: list[torch.Tensor],
    gt_disp: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, dict]:
    total = gt_disp.sum() * 0.0
    init_h, init_w = init_disp.shape[-2:]
    scale = init_w / float(gt_disp.shape[-1])
    init_gt = F.interpolate(gt_disp, size=(init_h, init_w), mode="nearest") * scale
    init_valid = F.interpolate(valid.float(), size=(init_h, init_w), mode="nearest") > 0.5
    init_loss = F.smooth_l1_loss(init_disp[init_valid], init_gt[init_valid]) if init_valid.any() else total

    disp_total = total * 0.0
    for index, prediction in enumerate(predictions):
        weight = gamma ** (len(predictions) - index - 1)
        finite = valid & torch.isfinite(prediction)
        if not finite.any():
            continue
        disp_loss = F.l1_loss(prediction[finite].float(), gt_disp[finite].float())
        disp_total = disp_total + weight * disp_loss
    total = init_loss + disp_total
    return total, {
        "loss": float(total.detach()),
        "loss_init_disp": float(init_loss.detach()),
        "loss_disp": float(disp_total.detach()),
    }


@torch.no_grad()
def final_metrics(prediction, gt_depth, gt_disp, valid, focal_baseline) -> dict:
    mask = valid & torch.isfinite(prediction) & (prediction > 0.0)
    count = int(mask.sum())
    if not count:
        return {"count": 0, "depth_abs": 0.0, "depth_sq": 0.0, "disp_abs": 0.0}
    pred_depth = focal_baseline / prediction[mask]
    depth_error = pred_depth - gt_depth[mask]
    disp_error = prediction[mask] - gt_disp[mask]
    return {
        "count": count,
        "depth_abs": float(depth_error.abs().double().sum()),
        "depth_sq": float(depth_error.double().square().sum()),
        "disp_abs": float(disp_error.abs().double().sum()),
    }


@torch.no_grad()
def sparse_lidar_depth_metrics(
    prediction_disp: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_valid: torch.Tensor,
    focal_baseline: float,
    min_depth: float,
    max_depth: float,
) -> dict:
    mask = (
        gt_valid.bool()
        & torch.isfinite(gt_depth)
        & (gt_depth >= min_depth)
        & (gt_depth <= max_depth)
    )
    count = int(mask.sum())
    if count == 0:
        return {
            "count": 0, "abs_sum": 0.0, "sq_sum": 0.0,
            "delta1": 0, "delta2": 0, "delta3": 0,
        }
    min_disp = float(focal_baseline) / float(max_depth)
    max_disp = float(focal_baseline) / float(min_depth)
    raw_disp = prediction_disp[mask].float()
    pred_disp = torch.nan_to_num(
        raw_disp, nan=min_disp, posinf=min_disp, neginf=min_disp,
    ).clamp(min=min_disp, max=max_disp)
    pred_depth = float(focal_baseline) / pred_disp
    target_depth = gt_depth[mask].float()
    error = pred_depth - target_depth
    ratio = torch.maximum(
        pred_depth / target_depth,
        target_depth / pred_depth.clamp(min=min_depth),
    )
    return {
        "count": count,
        "abs_sum": float(error.abs().double().sum()),
        "sq_sum": float(error.double().square().sum()),
        "delta1": int((ratio <= 1.25).sum()),
        "delta2": int((ratio <= 1.25 ** 2).sum()),
        "delta3": int((ratio <= 1.25 ** 3).sum()),
    }


def save_manifest(path: Path, dataset: RGBNIRDenseDepthDataset, indices: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# key\trgb\tnir\tdepth\n")
        for index in indices:
            rgb, nir, depth, key = dataset.samples[index]
            stream.write(f"{key}\t{rgb.resolve()}\t{nir.resolve()}\t{depth.resolve()}\n")


def to_yamlable(value):
    if isinstance(value, dict):
        return {str(key): to_yamlable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_yamlable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_yamlable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--nir-dir", type=Path, default=DEFAULT_NIR)
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--feature-lr", type=float,
        help="Optional learning rate for model.feature; other modules use --lr",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--valid-iters", type=int, default=22)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--eval-min-depth", type=float, default=0.1)
    parser.add_argument("--eval-max-depth", type=float, default=200.0)
    parser.add_argument("--sparse-depth", action="store_true", help="Use sparse LiDAR depth files and supervise only their nonzero pixels")
    parser.add_argument("--depth-suffix", type=str, help="Exact depth filename suffix, e.g. _depth_gt_sparse_u16.png")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs and batch-size must be >= 1; lr must be positive")
    if args.feature_lr is not None and args.feature_lr <= 0:
        parser.error("feature-lr must be positive when specified")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must be between 0 and 1")
    if args.eval_min_depth <= 0.0 or args.eval_max_depth <= args.eval_min_depth:
        parser.error("Require 0 < --eval-min-depth < --eval-max-depth")
    return args


def main(*, sparse_lidar_validation: bool = False) -> None:
    args = parse_args()
    seed_everything(args.seed)
    calibration = load_calibration(args.calibration_file)
    dataset = RGBNIRDenseDepthDataset(
        args.rgb_dir, args.nir_dir, args.depth_dir, calibration,
        max_disp=args.max_disp, max_depth=args.max_depth, sparse_depth=args.sparse_depth,
        depth_suffix=args.depth_suffix,
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit is not None:
        indices = indices[: min(args.limit, len(indices))]
    if len(indices) < 2:
        raise RuntimeError("At least two matched samples are required")
    val_count = min(len(indices) - 1, max(1, int(len(indices) * args.val_fraction)))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    args.output.mkdir(parents=True, exist_ok=True)
    save_manifest(args.output / "train.txt", dataset, train_indices)
    save_manifest(args.output / "val.txt", dataset, val_indices)

    sample = dataset[train_indices[0]]
    sample_valid_count = int(sample["valid"].sum())
    if not sample_valid_count:
        raise RuntimeError(f"First training sample has no valid supervised pixels: {sample['key']}")
    print(
        f"seed={args.seed}, train={len(train_indices)}, val={len(val_indices)}, "
        f"left={tuple(sample['left'].shape)}, right={tuple(sample['right'].shape)}, "
        f"valid_depth_pixels={sample_valid_count}, "
        f"depth_range_m=({sample['depth'][sample['valid']].min().item():.4f}, "
        f"{sample['depth'][sample['valid']].max().item():.4f}), "
        f"fx={calibration['fx']:.6f}, baseline_m={calibration['baseline']:.9f}",
        flush=True,
    )
    if args.check_data_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Fast-FoundationStereo fine-tuning requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU index {args.gpu}; available CUDA devices: {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    try:
        model = torch.load(args.model, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Missing model dependency '{error.name}'. Install this repository's "
            "requirements before training: pip install -r requirements.txt"
        ) from error
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.to(device).train()
    with (args.output / "cfg.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(to_yamlable(dict(model.args)), stream, sort_keys=False)
    if args.feature_lr is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        feature_params = list(model.feature.parameters())
        feature_param_ids = {id(param) for param in feature_params}
        stereo_params = [
            param for param in model.parameters() if id(param) not in feature_param_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": feature_params, "lr": args.feature_lr},
                {"params": stereo_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(
            f"Learning rates: feature={args.feature_lr:g}, "
            f"stereo_modules={args.lr:g}", flush=True,
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
        best_key = "best_val_depth_mae_m" if sparse_lidar_validation else "best_val"
        best_val = float(checkpoint.get(best_key, best_val))
        history = list(checkpoint.get("history", []))

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    focal_baseline = calibration["fx"] * calibration["baseline"]
    run_config = {
        **vars(args),
        "rgb_dir": str(args.rgb_dir), "nir_dir": str(args.nir_dir),
        "depth_dir": str(args.depth_dir), "model": str(args.model),
        "resume": None if args.resume is None else str(args.resume),
        "output": str(args.output), "calibration_file": str(args.calibration_file),
        "depth_scale_m_per_uint16": DEPTH_METRES_PER_UINT16,
        "fx_rectified": calibration["fx"], "baseline_m": calibration["baseline"],
        "supervision": (
            "rectified sparse depth converted to disparity; nonzero valid labels only; "
            "initial SmoothL1 plus gamma-weighted iterative L1 disparity loss"
            if args.sparse_depth
            else "rectified depth converted to disparity; nonzero valid labels only; "
            "initial SmoothL1 plus gamma-weighted iterative L1 disparity loss"
        ),
        "validation_metrics": (
            "MAE, RMSE, delta1, delta2, delta3 only at nonzero validation "
            "sparse-LiDAR ground-truth pixels"
            if sparse_lidar_validation else "legacy label-based depth/disparity metrics"
        ),
        "best_model_selection": (
            "minimum validation sparse-LiDAR depth MAE"
            if sparse_lidar_validation else "minimum validation loss"
        ),
    }
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for epoch in range(start_epoch, args.epochs):
        epoch_result = {"epoch": epoch + 1}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            loss_sum = 0.0
            batches = 0
            if sparse_lidar_validation:
                totals = {
                    "count": 0, "abs_sum": 0.0, "sq_sum": 0.0,
                    "delta1": 0, "delta2": 0, "delta3": 0,
                }
            else:
                totals = {"count": 0, "depth_abs": 0.0, "depth_sq": 0.0, "disp_abs": 0.0}
            context = torch.enable_grad() if phase == "train" else torch.no_grad()
            with context:
                for step, batch in enumerate(loader, start=1):
                    left = batch["left"].to(device, non_blocking=True)
                    right = batch["right"].to(device, non_blocking=True)
                    gt_depth = batch["depth"].to(device, non_blocking=True)
                    gt_disp = batch["disp"].to(device, non_blocking=True)
                    valid = batch["valid"].to(device, non_blocking=True).bool()
                    eval_depth = batch["eval_depth"].to(device, non_blocking=True)
                    eval_valid = batch["eval_valid"].to(device, non_blocking=True).bool()
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    init_disp, predictions = model(
                        left, right, iters=args.valid_iters, test_mode=False,
                        optimize_build_volume="pytorch1",
                    )
                    loss, details = sequence_loss(
                        init_disp, predictions, gt_disp, valid, args.gamma,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss: {details}; keys={batch['key']}")
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    if sparse_lidar_validation and phase == "val":
                        metrics = sparse_lidar_depth_metrics(
                            predictions[-1], eval_depth, eval_valid, focal_baseline,
                            args.eval_min_depth, args.eval_max_depth,
                        )
                        for name in totals:
                            totals[name] += metrics[name]
                    elif not sparse_lidar_validation:
                        metrics = final_metrics(
                            predictions[-1], gt_depth, gt_disp, valid, focal_baseline
                        )
                        for name in totals:
                            totals[name] += metrics[name]
                    loss_sum += float(loss.detach())
                    batches += 1
                    if step == 1 or step == len(loader) or step % 20 == 0:
                        print(
                            f"epoch {epoch + 1}/{args.epochs} {phase} {step}/{len(loader)} "
                            f"loss={details['loss']:.6f} init_disp={details['loss_init_disp']:.6f} "
                            f"iter_disp={details['loss_disp']:.6f}", flush=True,
                        )
            epoch_result[f"{phase}_loss"] = loss_sum / max(batches, 1)
            if sparse_lidar_validation and phase == "val":
                count = totals["count"]
                epoch_result["val_depth_mae_m"] = (
                    totals["abs_sum"] / count if count else None
                )
                epoch_result["val_depth_rmse_m"] = (
                    (totals["sq_sum"] / count) ** 0.5 if count else None
                )
                epoch_result["val_delta1"] = totals["delta1"] / count if count else None
                epoch_result["val_delta2"] = totals["delta2"] / count if count else None
                epoch_result["val_delta3"] = totals["delta3"] / count if count else None
            elif not sparse_lidar_validation:
                count = totals["count"]
                epoch_result[f"{phase}_depth_mae_m"] = totals["depth_abs"] / count if count else None
                epoch_result[f"{phase}_depth_rmse_m"] = (totals["depth_sq"] / count) ** 0.5 if count else None
                epoch_result[f"{phase}_disp_mae_px"] = totals["disp_abs"] / count if count else None
                epoch_result[f"{phase}_valid_pixels"] = count

        history.append(epoch_result)
        selection_metric = (
            epoch_result["val_depth_mae_m"]
            if sparse_lidar_validation else epoch_result["val_loss"]
        )
        if selection_metric is None:
            raise RuntimeError("Validation produced no valid sparse-LiDAR depth pixels")
        if selection_metric < best_val:
            best_val = selection_metric
            best_checkpoint = {
                "model": model.state_dict(), "epoch": epoch + 1,
                "val_loss": epoch_result["val_loss"], "run_config": run_config,
            }
            if sparse_lidar_validation:
                best_checkpoint["val_depth_mae_m"] = best_val
                best_checkpoint["best_val_depth_mae_m"] = best_val
            torch.save(best_checkpoint, args.output / "best_model.pt")
            # Same serialized format expected by run_demo.py/run_demo_rectified.py.
            torch.save(model, args.output / "model_best_serialize.pth")
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch + 1,
            "best_val": best_val, "history": history, "run_config": run_config,
        }
        if sparse_lidar_validation:
            checkpoint["best_val_depth_mae_m"] = best_val
        torch.save(checkpoint, args.output / "latest.pt")
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
