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
    parser.add_argument(
        "--sparse-gt-dir",
        type=Path,
        default=Path("/media/ddc/新加卷/hys/qmy/depth/bevneed/raw_left_sparse_depth_u16"),
        help="Raw distorted left-camera LiDAR depth, uint16 metres*256",
    )
    parser.add_argument("--eval-min-depth", type=float, default=0.1)
    parser.add_argument("--eval-max-depth", type=float, default=200.0)
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
    if args.eval_min_depth <= 0.0 or args.eval_max_depth <= args.eval_min_depth:
        parser.error("Require 0 < --eval-min-depth < --eval-max-depth")
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
        "K_left": k_left,
        "dist_left": d_left,
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


def rectify_sparse_left_depth(raw_depth_m: np.ndarray, calibration: dict) -> np.ndarray:
    """Forward-project raw distorted camera-Z samples onto the rectified grid."""
    expected_w, expected_h = calibration["size"]
    if raw_depth_m.shape != (expected_h, expected_w):
        raise ValueError(
            f"Raw sparse depth must be {expected_w}x{expected_h}, got {raw_depth_m.shape}"
        )
    valid = np.isfinite(raw_depth_m) & (raw_depth_m > 0.0)
    vv, uu = np.nonzero(valid)
    rectified = np.zeros((expected_h, expected_w), dtype=np.float32)
    if uu.size == 0:
        return rectified

    pixels = np.stack((uu, vv), axis=-1).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(
        pixels,
        calibration["K_left"],
        calibration["dist_left"],
    ).reshape(-1, 2)
    z_raw = raw_depth_m[vv, uu].astype(np.float64)
    points_raw = np.column_stack(
        (normalized[:, 0] * z_raw, normalized[:, 1] * z_raw, z_raw)
    )
    points_rectified = points_raw @ calibration["R1"].T
    z_rectified = points_rectified[:, 2]
    projected = points_rectified @ calibration["P1"][:, :3].T
    finite = (
        np.isfinite(projected).all(axis=1)
        & np.isfinite(z_rectified)
        & (z_rectified > 0.0)
    )
    projected = projected[finite]
    z_rectified = z_rectified[finite]
    u_rect = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
    v_rect = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
    inside = (
        (u_rect >= 0) & (u_rect < expected_w)
        & (v_rect >= 0) & (v_rect < expected_h)
    )
    u_rect, v_rect, z_rectified = (
        u_rect[inside], v_rect[inside], z_rectified[inside]
    )

    # Keep the nearest LiDAR return if multiple samples round to one pixel.
    flat = np.full(expected_h * expected_w, np.inf, dtype=np.float32)
    np.minimum.at(
        flat,
        v_rect * expected_w + u_rect,
        z_rectified.astype(np.float32),
    )
    flat[~np.isfinite(flat)] = 0.0
    return flat.reshape(expected_h, expected_w)


class RGBNIRPseudoDisparityDataset(Dataset):
    def __init__(
        self,
        rgb_dir: Path,
        nir_dir: Path,
        pseudo_root: Path,
        sparse_gt_dir: Path,
        calibration: dict,
        max_disp: float,
    ) -> None:
        self.calibration = calibration
        self.max_disp = float(max_disp)
        self.sparse_eval_indices: set[int] = set()
        rgb = index_files(rgb_dir, "_S0_rgb.png")
        nir = index_files(nir_dir, "_S0_nir.png")
        disparities = index_files(pseudo_root / "disparity_left_npy", "_disp_left_px.npy")
        masks = index_files(pseudo_root / "valid_mask", "_valid_mask.png")
        sparse_gt = index_files(sparse_gt_dir, "_depth_gt_sparse_u16.png")
        keys = sorted(set(rgb) & set(nir) & set(disparities) & set(masks) & set(sparse_gt))
        if not keys:
            raise RuntimeError(
                "No matching RGB/NIR/pseudo-disparity/mask/sparse-GT timestamp keys"
            )
        self.samples = [
            (rgb[k], nir[k], disparities[k], masks[k], sparse_gt[k], k) for k in keys
        ]
        print(
            "Dataset counts:",
            {
                "RGB": len(rgb),
                "NIR": len(nir),
                "disparity": len(disparities),
                "mask": len(masks),
                "sparse_gt": len(sparse_gt),
                "matched": len(keys),
            },
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def set_sparse_eval_indices(self, indices: list[int]) -> None:
        self.sparse_eval_indices = set(indices)

    def __getitem__(self, index: int) -> dict:
        rgb_path, nir_path, disparity_path, mask_path, sparse_gt_path, key = self.samples[index]
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
        sparse_gt_m = np.zeros((expected_h, expected_w), dtype=np.float32)
        if index in self.sparse_eval_indices:
            sparse_gt_u16 = cv2.imread(str(sparse_gt_path), cv2.IMREAD_UNCHANGED)
            if (
                sparse_gt_u16 is None
                or sparse_gt_u16.dtype != np.uint16
                or sparse_gt_u16.shape != (expected_h, expected_w)
            ):
                got = None if sparse_gt_u16 is None else (
                    sparse_gt_u16.dtype, sparse_gt_u16.shape
                )
                raise ValueError(
                    f"Sparse GT must be uint16 {expected_w}x{expected_h}: "
                    f"{sparse_gt_path}; got {got}"
                )
            sparse_gt_m = rectify_sparse_left_depth(
                sparse_gt_u16.astype(np.float32) / 256.0,
                self.calibration,
            )
        return {
            "left": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
            "right": torch.from_numpy(nir.transpose(2, 0, 1).copy()).float(),
            "disp": torch.from_numpy(disparity[None].copy()).float(),
            "valid": torch.from_numpy(valid[None].copy()),
            "sparse_gt_depth": torch.from_numpy(sparse_gt_m[None].copy()).float(),
            "key": key,
        }


def save_manifest(path: Path, dataset: RGBNIRPseudoDisparityDataset, indices: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# key\trgb\tnir\tdisparity\tmask\tsparse_gt_depth\n")
        for index in indices:
            rgb, nir, disparity, mask, sparse_gt, key = dataset.samples[index]
            stream.write(
                f"{key}\t{rgb.resolve()}\t{nir.resolve()}\t"
                f"{disparity.resolve()}\t{mask.resolve()}\t{sparse_gt.resolve()}\n"
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


@torch.no_grad()
def sparse_depth_metrics(
    prediction_disp: torch.Tensor,
    sparse_gt_depth: torch.Tensor,
    focal_baseline: float,
    min_depth: float,
    max_depth: float,
) -> dict:
    gt_valid = (
        torch.isfinite(sparse_gt_depth)
        & (sparse_gt_depth >= min_depth)
        & (sparse_gt_depth <= max_depth)
    )
    count = int(gt_valid.sum())
    if count == 0:
        return {
            "count": 0, "abs_sum": 0.0, "sq_sum": 0.0,
            "delta1": 0, "delta2": 0, "delta3": 0,
            "invalid_disp": 0, "clipped_disp": 0,
        }
    # Keep every real-GT pixel, but evaluate predictions only inside the same
    # metric-depth range. Invalid/non-positive disparities receive the maximum
    # depth penalty instead of exploding to tens of millions of metres.
    raw_disp = prediction_disp[gt_valid].float()
    min_disp = float(focal_baseline) / float(max_depth)
    max_disp = float(focal_baseline) / float(min_depth)
    finite = torch.isfinite(raw_disp)
    invalid_disp = (~finite) | (raw_disp <= 0.0)
    clipped_disp = (~finite) | (raw_disp < min_disp) | (raw_disp > max_disp)
    pred_disp = torch.nan_to_num(
        raw_disp, nan=min_disp, posinf=min_disp, neginf=min_disp
    ).clamp(min=min_disp, max=max_disp)
    pred_depth = float(focal_baseline) / pred_disp
    gt_depth = sparse_gt_depth[gt_valid].float()
    error = pred_depth - gt_depth
    ratio = torch.maximum(pred_depth / gt_depth, gt_depth / pred_depth.clamp(min=1e-6))
    return {
        "count": count,
        "abs_sum": float(error.abs().double().sum()),
        "sq_sum": float(error.double().square().sum()),
        "delta1": int((ratio <= 1.25).sum()),
        "delta2": int((ratio <= 1.25 ** 2).sum()),
        "delta3": int((ratio <= 1.25 ** 3).sum()),
        "invalid_disp": int(invalid_disp.sum()),
        "clipped_disp": int(clipped_disp.sum()),
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
        args.rgb_dir, args.nir_dir, args.pseudo_root, args.sparse_gt_dir,
        calibration, args.max_disp
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

    sample = dataset[train_indices[0]]
    valid_count = int(sample["valid"].sum())
    if valid_count == 0:
        raise RuntimeError(f"First training sample has no valid labels: {sample['key']}")
    valid_disp = sample["disp"][sample["valid"]]
    val_sample = dataset[val_indices[0]]
    sparse_gt_count = int(
        (
            (val_sample["sparse_gt_depth"] >= args.eval_min_depth)
            & (val_sample["sparse_gt_depth"] <= args.eval_max_depth)
        ).sum()
    )
    if sparse_gt_count == 0:
        raise RuntimeError(
            f"First validation sample has no valid sparse GT after rectification: "
            f"{val_sample['key']}"
        )
    print(
        f"seed={args.seed}, train={len(train_indices)}, val={len(val_indices)}, "
        f"left={tuple(sample['left'].shape)}, right={tuple(sample['right'].shape)}, "
        f"valid_pixels={valid_count}, disparity_range_px="
        f"({valid_disp.min().item():.4f}, {valid_disp.max().item():.4f}), "
        f"first_val_sparse_gt_pixels={sparse_gt_count}, "
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
        # Older checkpoints used pseudo-label val_loss for best_val. Do not
        # compare that quantity with the new sparse-LiDAR depth MAE.
        best_val = float(checkpoint.get("best_val_depth_mae_m", best_val))
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
        "sparse_gt_dir": str(args.sparse_gt_dir),
        "calibration_file": str(args.calibration_file),
        "model": str(args.model),
        "resume": None if args.resume is None else str(args.resume),
        "output": str(args.output),
        "rectification_alpha": alpha,
        "fx_rectified": calibration["fx"],
        "baseline_m": calibration["baseline"],
        "loss": "unchanged: initial SmoothL1 plus gamma-weighted iterative L1",
        "label": "rectified left pseudo disparity filtered by LR-consistency mask",
        "validation_metrics": (
            "predicted depth from final disparity compared with rectified raw LiDAR "
            "depth: MAE, RMSE, delta1, delta2, delta3"
        ),
        "best_model_selection": "minimum validation sparse-LiDAR depth MAE",
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
            disparity_totals = {"count": 0, "abs_sum": 0.0, "bad1": 0, "bad3": 0}
            depth_totals = {
                "count": 0,
                "abs_sum": 0.0,
                "sq_sum": 0.0,
                "delta1": 0,
                "delta2": 0,
                "delta3": 0,
                "invalid_disp": 0,
                "clipped_disp": 0,
            }
            context = torch.enable_grad() if phase == "train" else torch.no_grad()
            with context:
                for step, batch in enumerate(loader, start=1):
                    left = batch["left"].to(device, non_blocking=True)
                    right = batch["right"].to(device, non_blocking=True)
                    gt_disp = batch["disp"].to(device, non_blocking=True)
                    valid = batch["valid"].to(device, non_blocking=True).bool()
                    sparse_gt_depth = batch["sparse_gt_depth"].to(
                        device, non_blocking=True
                    )
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

                    if phase == "train":
                        metrics = disparity_metrics(predictions[-1], gt_disp, valid)
                        for name in disparity_totals:
                            disparity_totals[name] += metrics[name]
                    else:
                        metrics = sparse_depth_metrics(
                            predictions[-1],
                            sparse_gt_depth,
                            calibration["fx"] * calibration["baseline"],
                            args.eval_min_depth,
                            args.eval_max_depth,
                        )
                        for name in depth_totals:
                            depth_totals[name] += metrics[name]
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

            epoch_result[f"{phase}_loss"] = loss_sum / max(batches, 1)
            if phase == "train":
                count = disparity_totals["count"]
                epoch_result["train_disp_mae_px"] = (
                    disparity_totals["abs_sum"] / count if count else None
                )
                epoch_result["train_valid_pixels"] = count
            else:
                count = depth_totals["count"]
                epoch_result["val_depth_mae_m"] = (
                    depth_totals["abs_sum"] / count if count else None
                )
                epoch_result["val_depth_rmse_m"] = (
                    (depth_totals["sq_sum"] / count) ** 0.5 if count else None
                )
                epoch_result["val_delta1"] = (
                    depth_totals["delta1"] / count if count else None
                )
                epoch_result["val_delta2"] = (
                    depth_totals["delta2"] / count if count else None
                )
                epoch_result["val_delta3"] = (
                    depth_totals["delta3"] / count if count else None
                )
                epoch_result["val_sparse_gt_pixels"] = count
                epoch_result["val_invalid_disp_pixels"] = depth_totals["invalid_disp"]
                epoch_result["val_invalid_disp_rate"] = (
                    depth_totals["invalid_disp"] / count if count else None
                )
                epoch_result["val_clipped_disp_pixels"] = depth_totals["clipped_disp"]
                epoch_result["val_clipped_disp_rate"] = (
                    depth_totals["clipped_disp"] / count if count else None
                )

        history.append(epoch_result)
        selection_metric = epoch_result["val_depth_mae_m"]
        if selection_metric is None:
            raise RuntimeError("Validation produced no valid sparse-LiDAR depth pixels")
        if selection_metric < best_val:
            best_val = selection_metric
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": epoch_result["val_loss"],
                    "val_depth_mae_m": selection_metric,
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
                "best_val_depth_mae_m": best_val,
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
