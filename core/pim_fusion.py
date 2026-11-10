"""Haar-wavelet PIM fusion adapted from GatedRCCBStereo."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ) / 2.0
        self.register_buffer("weight", filters[:, None].repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        x = F.pad(x, (0, width % 2, 0, height % 2))
        bands = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=2, groups=channels)
        bands = bands.reshape(batch, channels, 4, *bands.shape[-2:])
        return bands[:, :, 0], torch.cat(tuple(bands[:, :, i] for i in (1, 2, 3)), dim=1)


class WaveletS0DoLPFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarDWT(channels)

        def gate(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, 2, 1),
            )

        def projection(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

        self.low_gate = gate(channels * 2)
        self.high_gate = gate(channels * 6)
        self.high_projection = projection(channels * 3)
        self.output_projection = projection(channels * 2)

    def forward(self, s0: torch.Tensor, dolp: torch.Tensor) -> torch.Tensor:
        low_s0, high_s0 = self.dwt(s0)
        low_dolp, high_dolp = self.dwt(dolp)
        low_weights = self.low_gate(torch.cat((low_s0, low_dolp), dim=1)).softmax(dim=1)
        high_weights = self.high_gate(torch.cat((high_s0, high_dolp), dim=1)).softmax(dim=1)
        low = low_s0 * low_weights[:, :1] + low_dolp * low_weights[:, 1:]
        high = self.high_projection(
            high_s0 * high_weights[:, :1] + high_dolp * high_weights[:, 1:]
        )
        low = F.interpolate(low, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        high = F.interpolate(high, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        return self.output_projection(torch.cat((low, high), dim=1))


class WaveletPIM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1_s0 = nn.BatchNorm2d(channels)
        self.norm1_dolp = nn.BatchNorm2d(channels)
        self.norm2_s0 = nn.BatchNorm2d(channels)
        self.norm2_dolp = nn.BatchNorm2d(channels)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.mlp_s0 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.mlp_dolp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.wavelet_fusion = WaveletS0DoLPFusion(channels)
        self.layer_scale_1_s0 = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_1_dolp = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_2_s0 = nn.Parameter(torch.full((channels,), 1e-5))
        self.layer_scale_2_dolp = nn.Parameter(torch.full((channels,), 1e-5))
        self.fusion_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def scale(gamma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return gamma.view(1, -1, 1, 1) * x

    def forward(self, s0: torch.Tensor, dolp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_s0 = self.norm1_s0(s0)
        normalized_dolp = self.norm1_dolp(dolp)
        s0 = s0 + self.scale(self.layer_scale_1_s0, self.pool(normalized_s0) - normalized_s0)
        dolp = dolp + self.scale(
            self.layer_scale_1_dolp, self.pool(normalized_dolp) - normalized_dolp
        )
        normalized_s0 = self.norm2_s0(s0)
        normalized_dolp = self.norm2_dolp(dolp)
        fusion = self.fusion_scale * self.wavelet_fusion(normalized_s0, normalized_dolp)
        s0 = s0 + self.scale(self.layer_scale_2_s0, self.mlp_s0(normalized_s0))
        dolp = dolp + self.scale(self.layer_scale_2_dolp, self.mlp_dolp(normalized_dolp))
        return s0 + fusion, dolp + fusion


class DOLPPIMFeatureFusion(nn.Module):
    def __init__(self, feature_channels: int):
        super().__init__()
        self.pim = WaveletPIM(feature_channels)
        self.output_projection = nn.Conv2d(
            2 * feature_channels, feature_channels, 1, bias=False
        )
        with torch.no_grad():
            self.output_projection.weight.zero_()
            channels = torch.arange(feature_channels)
            self.output_projection.weight[channels, channels, 0, 0] = 0.5
            self.output_projection.weight[
                channels, channels + feature_channels, 0, 0
            ] = 0.5

    def forward(self, image_feature: torch.Tensor, dolp_feature: torch.Tensor) -> torch.Tensor:
        image_feature, dolp_feature = self.pim(image_feature, dolp_feature)
        return self.output_projection(torch.cat((image_feature, dolp_feature), dim=1))
