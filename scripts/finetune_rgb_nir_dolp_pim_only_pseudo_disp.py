"""Train only the left/right DoLP-PIM fusion modules of frozen FFS."""

from __future__ import annotations

import finetune_rgb_nir_dolp_pim_pseudo_disp as trainer


if __name__ == "__main__":
    trainer.DEFAULT_OUTPUT = (
        trainer.ROOT / "finetune_rgb_nir_dolp_pim_only_pseudo_disp"
    )
    trainer.main(train_pim_only=True)
