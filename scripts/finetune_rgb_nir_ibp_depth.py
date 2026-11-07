"""Fine-tune FFS using IBP-generated sparse depth labels (uint16 metres * 256)."""

from __future__ import annotations

import sys
from pathlib import Path

from finetune_rgb_nir_depth import main


ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = [
    "--rgb-dir", "/media/ddc/新加卷/hys/qmy/depth/data/RGBS0",
    "--nir-dir", "/media/ddc/新加卷/hys/qmy/depth/data/NIRS0",
    "--depth-dir", "/media/ddc/新加卷/hys/qmy/depth/data/IBPdepth",
    "--calibration-file", str(ROOT / "demo_data/rgb_nir_stereo_calibration.json"),
    "--model", str(ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"),
    "--output", str(ROOT / "finetune_rgb_nir_ibp_depth"),
    "--sparse-depth",
    "--depth-suffix", "_depth_gt_sparse_u16.png",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULTS, *sys.argv[1:]]
    main()
