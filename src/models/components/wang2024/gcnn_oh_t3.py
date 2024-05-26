from gconv.gnn import (
    GSeparableConvSE3,
    GLiftingConvSE3,
    GReLU,
    GBatchNorm3d,
    GAvgGroupPool
)
from gconv.geometry.groups import so3
from torch import Tensor
import torch.nn as nn
from src.models.components.wang2024.upsample_utils import GroupUpsampler3D


class GUpsamplingBlock(nn.Module):
    """Upsampling block for gcnn network"""

    def __init__(self, hidden_dim: int, kernel_size: int, grid_H: Tensor):
        super().__init__()
        self.upsampler = GroupUpsampler3D(nn.Upsample(scale_factor=2, mode="trilinear"))
        conv_kwargs = {
            "in_channels": hidden_dim,
            "out_channels": hidden_dim,
            "kernel_size": kernel_size,
            "permute_output_grid": False,
            "grid_H": grid_H,
            "mask": False,
            "padding": kernel_size // 2,
        }
        self.upconv = GSeparableConvSE3(
            **conv_kwargs,
            conv_mode="3d_transposed",
            output_padding=1,
            stride=2,
        )
        self.bn1 = GBatchNorm3d(hidden_dim)
        self.relu1 = GReLU()
        self.conv = GSeparableConvSE3(**conv_kwargs)
        self.bn2 = GBatchNorm3d(hidden_dim)
        self.relu2 = GReLU()

    def forward(self, x: Tensor, H: Tensor) -> tuple[Tensor, Tensor]:
        z = x.clone()
        z, H = self.upconv(z, H)
        z, H = self.bn1(z, H)
        z, H = self.relu1(z, H)
        z, H = self.conv(z, H)
        z, H = self.bn2(z, H)
        z += self.upsampler(x)
        z, H = self.relu2(z, H)  # relu after skip connection
        return z, H


class GConvBlock(nn.Module):
    """Convolutional block for gcnn network"""

    def __init__(self, hidden_dim: int, kernel_size: int, grid_H: Tensor):
        super().__init__()
        conv_kwargs = {
            "in_channels": hidden_dim,
            "out_channels": hidden_dim,
            "kernel_size": kernel_size,
            "permute_output_grid": False,
            "grid_H": grid_H,
            "mask": False,
            "padding": kernel_size // 2,
        }
        self.conv1 = GSeparableConvSE3(**conv_kwargs)
        self.bn1 = GBatchNorm3d(hidden_dim)
        self.relu1 = GReLU()
        self.conv2 = GSeparableConvSE3(**conv_kwargs)
        self.bn2 = GBatchNorm3d(hidden_dim)
        self.relu2 = GReLU()

    def forward(self, x: Tensor, H: Tensor) -> tuple[Tensor, Tensor]:
        z = x.clone()
        z, H = self.conv1(z, H)
        z, H = self.bn1(z, H)
        z, H = self.relu1(z, H)
        z, H = self.conv2(z, H)
        z, H = self.bn2(z, H)
        z += x
        z, H = self.relu2(z, H)
        return z, H


class GCNNOhT3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.register_buffer("grid_Oh", so3.quat_to_matrix(so3.octahedral()))
        self.upsampler = GroupUpsampler3D(nn.Upsample(scale_factor=2, mode="trilinear"))
        conv_kwargs = {
            "kernel_size": kernel_size,
            "permute_output_grid": False,
            "grid_H": self.grid_Oh,
            "mask": False,
            "padding": kernel_size // 2,
        }
        self.lift = GLiftingConvSE3(
            in_channels=in_channels, out_channels=hidden_dim, **conv_kwargs
        )
        self.upconv_block1 = GUpsamplingBlock(hidden_dim, kernel_size, self.grid_Oh)
        self.upconv_block2 = GUpsamplingBlock(hidden_dim, kernel_size, self.grid_Oh)
        self.conv_block1 = GConvBlock(hidden_dim, kernel_size, self.grid_Oh)
        self.conv_block2 = GConvBlock(hidden_dim, kernel_size, self.grid_Oh)
        self.final_conv = GSeparableConvSE3(
            in_channels=hidden_dim, out_channels=out_channels, **conv_kwargs
        )
        self.pool = GAvgGroupPool()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, H, W]
        """
        z, H = self.lift(x, self.grid_Oh)
        z, H = self.upconv_block1(z, H)
        z, H = self.conv_block1(z, H)
        z, H = self.upconv_block2(z, H)
        z, H = self.conv_block2(z, H)
        z, H = self.final_conv(z, H)
        z = self.pool(z)
        return z

