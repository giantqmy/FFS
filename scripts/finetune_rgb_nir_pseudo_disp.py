"""Fine-tune Fast-FoundationStereo with LR-consistent pseudo disparity labels.

RGB and NIR images are rectified at load time. Pseudo disparities and masks
must already be on that rectified left-image grid; they are never remapped.
The loss is intentionally identical to finetune_rgb_nir_depth.py.
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
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from finetune_rgb_nir_depth import seed_everything, sequence_loss, to_yamlable  # noqa: E402


DEFAULT_DATA_ROOT = Path("/media/ddc/新加卷/hys/qmy/depth/data")
DEFAULT_MODEL = ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_CALIBRATION = ROOT / "demo_data" / "rgb_nir_stereo_calibration.json"
DEFAULT_OUTPUT = ROOT / "finetune_rgb_nir_pseudo_disp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_DATA_ROOT / "RGBS0")
    parser.add_argument("--nir-dir", type=Path, default=DEFAULT_DATA_ROOT / "NIRS0")
    parser.add_argument(
        "--pseudo-root",
        type=Path,
        required=True,
        help="Directory containing disparity_left_npy, valid_mask, and metadata.json",
    )
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--alpha",
        type=float,
        help="Rectification alpha override; default reads pseudo-root/metadata.json",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--feature-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--valid-iters", type=int, default=22)
    parser.add_argument("--max-disp", type=float, default=192.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0.0:
        parser.error("epochs and batch-size must be >= 1; lr must be positive")
    if args.feature_lr is not None and args.feature_lr <= 0.0:
        parser.error("--feature-lr must be positive")
    if args.max_disp <= 0.0:
        parser.error("--max-disp must be positive")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0, 1]")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    return args


def read_pseudo_metadata(pseudo_root: Path) -> dict:
    path = pseudo_root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pseudo-label metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration(path: Path, alpha: float) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = map(int, data["image_size"])
    k_left = np.ascontiguousarray(data["K_left"], dtype=np.float64).reshape(3, 3)
    k_right = np.ascontiguousarray(data["K_right"], dtype=np.float64).reshape(3, 3)
    d_left = np.ascontiguousarray(data["dist_left"], dtype=np.float64).reshape(1, -1)
    d_right = np.ascontiguousarray(data["dist_right"], dtype=np.float64).reshape(1, -1)
    rotation = np.ascontiguousarray(data["R_left_to_right"], dtype=np.float64).reshape(3, 3)
    translation = np.ascontiguousarray(data["T_left_to_right_m"], dtype=np.float64).reshape(3, 1)
    r1, r2, p1, p2, _, roi1, roi2 = cv2.stereoRectify(
        k_left,
        d_left,
        k_right,
        d_right,
        (width, height),
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    map_left = cv2.initUndistortRectifyMap(
        k_left, d_left, r1, p1, (width, height), cv2.CV_32FC1
    )
    map_right = cv2.initUndistortRectifyMap(
        k_right, d_right, r2, p2, (width, height), cv2.CV_32FC1
    )
    return {
        "size": (width, height),
        "map_left": map_left,
        "map_right": map_right,
        "P1": p1,
        "P2": p2,
        "R1": r1,
        "R2": r2,
        "roi_left": roi1,
        "roi_right": roi2,
        "fx": float(p1[0, 0]),
        "baseline": abs(float(p2[0, 3] / p2[0, 0])),
        "alpha": float(alpha),
    }


def verify_pseudo_geometry(metadata: dict, calibration: dict) -> None:
    required = ("P_left", "P_right")
    missing = [name for name in required if name not in metadata]
    if missing:
        raise KeyError(f"Pseudo metadata lacks geometry fields: {missing}")
    pseudo_p1 = np.asarray(metadata["P_left"], dtype=np.float64).reshape(3, 4)
    pseudo_p2 = np.asarray(metadata["P_right"], dtype=np.float64).reshape(3, 4)
    if not np.allclose(pseudo_p1, calibration["P1"], rtol=1e-6, atol=1e-3):
        difference = float(np.max(np.abs(pseudo_p1 - calibration["P1"])))
        raise ValueError(
            "Pseudo disparity P_left does not match image rectification "
            f"(max difference {difference:.6g}). Use the same --alpha and calibration."
        )
    if not np.allclose(pseudo_p2, calibration["P2"], rtol=1e-6, atol=1e-3):
        difference = float(np.max(np.abs(pseudo_p2 - calibration["P2"])))
        raise ValueError(
            "Pseudo disparity P_right does not match image rectification "
            f"(max difference {difference:.6g}). Use the same --alpha and calibration."
        )


def index_files(directory: Path, suffix: str) -> dict[str, Path]:
    return {path.name[: -len(suffix)]: path for path in directory.glob(f"*{suffix}")}


class RGBNIRPseudoDisparityDataset(Dataset):
    def __init__(
        self,
        rgb_dir: Path,
        nir_dir: Path,
        pseudo_root: Path,
        calibration: dict,
        max_disp: float,
    ) -> None:
        self.calibration = calibration
        self.max_disp = float(max_disp)
        rgb = index_files(rgb_dir, "_S0_rgb.png")
        nir = index_files(nir_dir, "_S0_nir.png")
        disparities = index_files(pseudo_root / "disparity_left_npy", "_disp_left_px.npy")
        masks = index_files(pseudo_root / "valid_mask", "_valid_mask.png")
        keys = sorted(set(rgb) & set(nir) & set(disparities) & set(masks))
        if not keys:
            raise RuntimeError("No matching RGB/NIR/pseudo-disparity/mask timestamp keys")
        self.samples = [(rgb[k], nir[k], disparities[k], masks[k], k) for k in keys]
        print(
            "Dataset counts:",
            {
                "RGB": len(rgb),
                "NIR": len(nir),
                "disparity": len(disparities),
                "mask": len(masks),
                "matched": len(keys),
            },
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        rgb_path, nir_path, disparity_path, mask_path, key = self.samples[index]
        rgb = imageio.imread(rgb_path)
        nir = imageio.imread(nir_path)
        disparity = np.load(disparity_path, allow_pickle=False)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        expected_w, expected_h = self.calibration["size"]

        if rgb.shape[:2] != (expected_h, expected_w):
            raise ValueError(f"Wrong RGB size for {rgb_path}: {rgb.shape}")
        if nir.shape[:2] != (expected_h, expected_w):
            raise ValueError(f"Wrong NIR size for {nir_path}: {nir.shape}")
        if disparity.shape != (expected_h, expected_w):
            raise ValueError(f"Wrong disparity shape for {disparity_path}: {disparity.shape}")
        if mask is None or mask.shape != (expected_h, expected_w):
            got = None if mask is None else mask.shape
            raise ValueError(f"Wrong mask shape for {mask_path}: {got}")

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
        nir = np.repeat(nir[..., None], 3, axis=2)

        disparity = np.asarray(disparity, dtype=np.float32)
        xx = np.arange(expected_w, dtype=np.float32)[None, :]
        valid = (
            (mask > 0)
            & np.isfinite(disparity)
            & (disparity > 0.0)
            & (disparity < self.max_disp)
            & ((xx - disparity) >= 0.0)
        )
        disparity = np.where(valid, disparity, 0.0).astype(np.float32)
        return {
            "left": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
            "right": torch.from_numpy(nir.transpose(2, 0, 1).copy()).float(),
            "disp": torch.from_numpy(disparity[None].copy()).float(),
            "valid": torch.from_numpy(valid[None].copy()),
            "key": key,
        }


def save_manifest(path: Path, dataset: RGBNIRPseudoDisparityDataset, indices: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# key\trgb\tnir\tdisparity\tmask\n")
        for index in indices:
            rgb, nir, disparity, mask, key = dataset.samples[index]
            stream.write(
                f"{key}\t{rgb.resolve()}\t{nir.resolve()}\t"
                f"{disparity.resolve()}\t{mask.resolve()}\n"
            )


@torch.no_grad()
def disparity_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    mask = valid & torch.isfinite(prediction)
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "abs_sum": 0.0, "bad1": 0, "bad3": 0}
    error = (prediction[mask].float() - target[mask].float()).abs()
    return {
        "count": count,
        "abs_sum": float(error.double().sum()),
        "bad1": int((error > 1.0).sum()),
        "bad3": int((error > 3.0).sum()),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    metadata = read_pseudo_metadata(args.pseudo_root)
    metadata_alpha = float(metadata["alpha"])
    alpha = metadata_alpha if args.alpha is None else float(args.alpha)
    if args.alpha is not None and not np.isclose(alpha, metadata_alpha, atol=1e-12):
        raise ValueError(
            f"--alpha={alpha} conflicts with pseudo metadata alpha={metadata_alpha}; "
            "labels and images must use the same rectification grid"
        )
    calibration = load_calibration(args.calibration_file, alpha)
    verify_pseudo_geometry(metadata, calibration)
    dataset = RGBNIRPseudoDisparityDataset(
        args.rgb_dir, args.nir_dir, args.pseudo_root, calibration, args.max_disp
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
    valid_count = int(sample["valid"].sum())
    if valid_count == 0:
        raise RuntimeError(f"First training sample has no valid labels: {sample['key']}")
    valid_disp = sample["disp"][sample["valid"]]
    print(
        f"seed={args.seed}, train={len(train_indices)}, val={len(val_indices)}, "
        f"left={tuple(sample['left'].shape)}, right={tuple(sample['right'].shape)}, "
        f"valid_pixels={valid_count}, disparity_range_px="
        f"({valid_disp.min().item():.4f}, {valid_disp.max().item():.4f}), "
        f"alpha={alpha}, fx={calibration['fx']:.6f}, "
        f"baseline_m={calibration['baseline']:.9f}",
        flush=True,
    )
    if args.check_data_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Fast-FoundationStereo fine-tuning requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}; CUDA device count={torch.cuda.device_count()}")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    try:
        model = torch.load(args.model, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Missing model dependency '{error.name}'. Install this repository's requirements."
        ) from error
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = int(args.max_disp)
    model.to(device).train()

    if args.feature_lr is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        feature_params = list(model.feature.parameters())
        feature_ids = {id(parameter) for parameter in feature_params}
        stereo_params = [
            parameter for parameter in model.parameters() if id(parameter) not in feature_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": feature_params, "lr": args.feature_lr},
                {"params": stereo_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(
            f"Learning rates: feature={args.feature_lr:g}, stereo_modules={args.lr:g}",
            flush=True,
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
        best_val = float(checkpoint.get("best_val", best_val))
        history = list(checkpoint.get("history", []))

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    run_config = {
        **vars(args),
        "rgb_dir": str(args.rgb_dir),
        "nir_dir": str(args.nir_dir),
        "pseudo_root": str(args.pseudo_root),
        "calibration_file": str(args.calibration_file),
        "model": str(args.model),
        "resume": None if args.resume is None else str(args.resume),
        "output": str(args.output),
        "rectification_alpha": alpha,
        "fx_rectified": calibration["fx"],
        "baseline_m": calibration["baseline"],
        "loss": "unchanged: initial SmoothL1 plus gamma-weighted iterative L1",
        "label": "rectified left pseudo disparity filtered by LR-consistency mask",
    }
    (args.output / "run_config.json").write_text(
        json.dumps(to_yamlable(run_config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "cfg.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(to_yamlable(dict(model.args)), stream, sort_keys=False)

    for epoch in range(start_epoch, args.epochs):
        epoch_result = {"epoch": epoch + 1}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            loss_sum = 0.0
            batches = 0
            totals = {"count": 0, "abs_sum": 0.0, "bad1": 0, "bad3": 0}
            context = torch.enable_grad() if phase == "train" else torch.no_grad()
            with context:
                for step, batch in enumerate(loader, start=1):
                    left = batch["left"].to(device, non_blocking=True)
                    right = batch["right"].to(device, non_blocking=True)
                    gt_disp = batch["disp"].to(device, non_blocking=True)
                    valid = batch["valid"].to(device, non_blocking=True).bool()
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    init_disp, predictions = model(
                        left,
                        right,
                        iters=args.valid_iters,
                        test_mode=False,
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
                    for name in totals:
                        totals[name] += metrics[name]
                    loss_sum += float(loss.detach())
                    batches += 1
                    if step == 1 or step == len(loader) or step % 20 == 0:
                        print(
                            f"epoch {epoch + 1}/{args.epochs} {phase} {step}/{len(loader)} "
                            f"loss={details['loss']:.6f} "
                            f"init_disp={details['loss_init_disp']:.6f} "
                            f"iter_disp={details['loss_disp']:.6f}",
                            flush=True,
                        )

            count = totals["count"]
            epoch_result[f"{phase}_loss"] = loss_sum / max(batches, 1)
            epoch_result[f"{phase}_disp_mae_px"] = totals["abs_sum"] / count if count else None
            epoch_result[f"{phase}_bad1"] = totals["bad1"] / count if count else None
            epoch_result[f"{phase}_bad3"] = totals["bad3"] / count if count else None
            epoch_result[f"{phase}_valid_pixels"] = count

        history.append(epoch_result)
        if epoch_result["val_loss"] < best_val:
            best_val = epoch_result["val_loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": best_val,
                    "run_config": run_config,
                },
                args.output / "best_model.pt",
            )
            torch.save(model, args.output / "model_best_serialize.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_val": best_val,
                "history": history,
                "run_config": run_config,
            },
            args.output / "latest.pt",
        )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
