"""DoLP-PIM adapter for a pretrained Fast-FoundationStereo model."""

from __future__ import annotations

import copy

import torch
from torch import nn

from core.foundation_stereo import normalize_image
from core.pim_fusion import DOLPPIMFeatureFusion


class PIMFeatureAdapter(nn.Module):
    """Fuse left/right S0 and DoLP at FFS's native 1/4 feature scale."""

    def __init__(self, primary_feature: nn.Module):
        super().__init__()
        self.primary_feature = primary_feature
        self.dolp_feature_left = copy.deepcopy(primary_feature)
        self.dolp_feature_right = copy.deepcopy(primary_feature)
        channels = int(primary_feature.d_out[0])
        self.fusion_left = DOLPPIMFeatureFusion(channels)
        self.fusion_right = DOLPPIMFeatureFusion(channels)
        self.d_out = primary_feature.d_out
        self._dolp_left: torch.Tensor | None = None
        self._dolp_right: torch.Tensor | None = None

    def set_auxiliary(self, left: torch.Tensor, right: torch.Tensor) -> None:
        if left.ndim != 4 or right.ndim != 4 or left.shape[1] != 3 or right.shape[1] != 3:
            raise ValueError(
                f"DoLP inputs must be Bx3xHxW; got left={tuple(left.shape)}, "
                f"right={tuple(right.shape)}"
            )
        self._dolp_left = left
        self._dolp_right = right

    def clear_auxiliary(self) -> None:
        self._dolp_left = None
        self._dolp_right = None

    def forward(self, primary_pair: torch.Tensor) -> list[torch.Tensor]:
        if self._dolp_left is None or self._dolp_right is None:
            raise RuntimeError("PIM auxiliary DoLP inputs were not set before FFS forward")
        batch = self._dolp_left.shape[0]
        if primary_pair.shape[0] != 2 * batch:
            raise ValueError(
                f"Expected primary pair batch {2 * batch}, got {primary_pair.shape[0]}"
            )
        primary = self.primary_feature(primary_pair)
        left_dolp = self.dolp_feature_left(normalize_image(self._dolp_left))
        right_dolp = self.dolp_feature_right(normalize_image(self._dolp_right))
        left_primary = primary[0][:batch]
        right_primary = primary[0][batch:]
        if left_primary.shape != left_dolp[0].shape or right_primary.shape != right_dolp[0].shape:
            raise ValueError(
                "S0 and DoLP feature shapes differ: "
                f"left={tuple(left_primary.shape)}/{tuple(left_dolp[0].shape)}, "
                f"right={tuple(right_primary.shape)}/{tuple(right_dolp[0].shape)}"
            )
        fused_x4 = torch.cat(
            (
                self.fusion_left(left_primary, left_dolp[0]),
                self.fusion_right(right_primary, right_dolp[0]),
            ),
            dim=0,
        )
        return [fused_x4, *primary[1:]]


class FastFoundationStereoPIM(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.base_model.feature = PIMFeatureAdapter(self.base_model.feature)

    @property
    def args(self):
        return self.base_model.args

    def forward(
        self,
        image_left: torch.Tensor,
        image_right: torch.Tensor,
        dolp_left: torch.Tensor,
        dolp_right: torch.Tensor,
        **kwargs,
    ):
        adapter = self.base_model.feature
        adapter.set_auxiliary(dolp_left, dolp_right)
        try:
            return self.base_model(image_left, image_right, **kwargs)
        finally:
            adapter.clear_auxiliary()
