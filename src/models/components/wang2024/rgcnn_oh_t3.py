from gconv.gnn import (
    RGLiftingConvSE3,
    RGSeparableConvSE3,
    GReLU,
    GBatchNorm3d,
    GAvgGroupPool,
)
from gconv.geometry.groups import so3
from torch import Tensor, nn
from src.models.components.wang2024.upsample_utils import GroupUpsampler3D


class RGUpsamplingBlock(nn.Module):
    """Upsampling block for rgcnn network"""

    def __init__(
        self, hidden_dim: int, kernel_size: int, num_filter_banks: int, grid_H: Tensor
    ):
        super().__init__()
        self.upsampler = GroupUpsampler3D(nn.Upsample(scale_factor=2, mode="trilinear"))
        conv_kwargs = {
            "in_channels": hidden_dim,
            "out_channels": hidden_dim,
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "permute_output_grid": False,
            "num_filter_banks": num_filter_banks,
            "grid_H": grid_H,
            "mask": False,
        }
        self.upconv = RGSeparableConvSE3(
            **conv_kwargs,
            conv_mode="3d_transposed",
            output_padding=1,
            stride=2,
        )
        self.bn1 = GBatchNorm3d(hidden_dim)
        self.relu1 = GReLU()
        self.conv = RGSeparableConvSE3(**conv_kwargs)
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
        z, H = self.relu2(z, H)
        return z, H


class RGConvBlock(nn.Module):
    """Convolutional block for rgcnn network"""

    def __init__(
        self, hidden_dim: int, kernel_size: int, num_filter_banks: int, grid_H: Tensor
    ):
        super().__init__()
        conv_kwargs = {
            "in_channels": hidden_dim,
            "out_channels": hidden_dim,
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "permute_output_grid": False,
            "num_filter_banks": num_filter_banks,
            "grid_H": grid_H,
            "mask": False,
        }
        self.conv1 = RGSeparableConvSE3(**conv_kwargs)
        self.bn1 = GBatchNorm3d(hidden_dim)
        self.relu1 = GReLU()
        self.conv2 = RGSeparableConvSE3(**conv_kwargs)
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


class RGCNNOhT3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filter_banks: int,
        kernel_size: int,
        hidden_dim: int,
    ):

        super().__init__()
        self.register_buffer("grid_Oh", so3.quat_to_matrix(so3.octahedral()))

        self.upsampler = GroupUpsampler3D(nn.Upsample(scale_factor=2, mode="trilinear"))

        self.lift = RGLiftingConvSE3(
            in_channels=in_channels,
            out_channels=hidden_dim,
            num_filter_banks=num_filter_banks,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            grid_H=self.grid_Oh,
            permute_output_grid=False,
            mask=False,
        )
        self.upconv_block1 = RGUpsamplingBlock(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_filter_banks=num_filter_banks,
            grid_H=self.grid_Oh,
        )
        self.upconv_block2 = RGUpsamplingBlock(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_filter_banks=num_filter_banks,
            grid_H=self.grid_Oh,
        )
        self.conv_block1 = RGConvBlock(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_filter_banks=num_filter_banks,
            grid_H=self.grid_Oh,
        )
        self.conv_block2 = RGConvBlock(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_filter_banks=num_filter_banks,
            grid_H=self.grid_Oh,
        )
        self.final_conv = RGSeparableConvSE3(
            in_channels=hidden_dim,
            out_channels=out_channels,
            num_filter_banks=num_filter_banks,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            permute_output_grid=False,
            grid_H=self.grid_Oh,
            mask=False,
        )
        self.final_conv = RGSeparableConvSE3(
            in_channels=hidden_dim,
            out_channels=out_channels,
            num_filter_banks=num_filter_banks,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            permute_output_grid=False,
            grid_H=self.grid_Oh,
            mask=False,
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
