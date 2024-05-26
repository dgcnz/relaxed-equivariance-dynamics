from torch import Tensor
import torch.nn as nn


class UpsamplingBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode="trilinear")
        conv_kwargs = {
            "in_channels": hidden_dim,
            "out_channels": hidden_dim,
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "dilation": 1,
        }
        self.upconv = nn.ConvTranspose3d(
            **conv_kwargs,
            stride=2,
            output_padding=1,
        )
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv = nn.Conv3d(
            **conv_kwargs,
            stride=1,
        )
        self.bn2 = nn.BatchNorm3d(hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        z = x.clone()
        z = self.upconv(z)
        z = self.bn1(z)
        z = self.relu1(z)
        z = self.conv(z)
        z = self.bn2(z)
        z += self.upsampler(x)
        z = self.relu2(z)
        return z


class ConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int):
        super().__init__()
        conv_kwargs = {
            "in_channels": hidden_dim,
            "out_channels": hidden_dim,
            "kernel_size": kernel_size,
            "padding": kernel_size // 2,
            "dilation": 1,
            "stride": 1,
        }
        self.conv1 = nn.Conv3d(**conv_kwargs)
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(**conv_kwargs)
        self.bn2 = nn.BatchNorm3d(hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        z = x.clone()
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z += x
        z = self.relu2(z)
        return z


class SuperResCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,
    ):

        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2, mode="trilinear")

        self.first_conv = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
        )
        self.upconv_block1 = UpsamplingBlock(hidden_dim, kernel_size)
        self.conv_block1 = ConvBlock(hidden_dim, kernel_size)
        self.upconv_block2 = UpsamplingBlock(hidden_dim, kernel_size)
        self.conv_block2 = ConvBlock(hidden_dim, kernel_size)
        self.final_conv = nn.Conv3d(
            hidden_dim,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.first_conv(x)
        z = self.upconv_block1(z)
        z = self.conv_block1(z)
        z = self.upconv_block2(z)
        z = self.conv_block2(z)
        z = self.final_conv(z)
        return z
