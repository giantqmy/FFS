"""Train RGB/NIR FFS with local-consistency depth and evaluate on sparse LiDAR."""

from __future__ import annotations

import sys

from finetune_rgb_nir_depth import main


DEFAULTS = [
    "--depth-suffix", "_depth_dense_u16.png",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *DEFAULTS, *sys.argv[1:]]
    main(
        sparse_lidar_validation=True,
        require_independent_sparse_gt=True,
    )
