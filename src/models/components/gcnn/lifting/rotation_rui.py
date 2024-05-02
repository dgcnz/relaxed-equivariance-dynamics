# Taken as is from: https://github.com/Rui1521/Equivariant-CNNs-Tutorial/tree/main
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.image_utils import rot_img


class RuiCNLiftingConvolution(nn.Module):
    """Lifting Convolution Layer for finite rotation group.

    Attributes:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size
        group_order: the order of rotation groups
        activation: whether to use relu.
    """

    def __init__(self, in_channels, out_channels, kernel_size, group_order, activation=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation

        # Initialize an unconstrained kernel.
        self.kernel = torch.nn.Parameter(
            torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )

        # Initialize weights
        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))

    def generate_filter_bank(self):
        # Obtain a stack of rotated filters
        # Rotate kernels by 0, 90, 180, and 270 degrees
        # ==============================
        filter_bank = torch.stack(
            [
                rot_img(self.kernel, -np.pi * 2 / self.group_order * i)
                for i in range(self.group_order)
            ]
        )
        # ==============================

        # [#out, group_order, #in, k, k]
        filter_bank = filter_bank.transpose(0, 1)
        return filter_bank

    def forward(self, x):
        # input shape: [bz, #in, h, w]
        # output shape: [bz, #out, group order, h, w]

        # generate filter bank given input group order
        filter_bank = self.generate_filter_bank()

        # concatenate the first two dims before convolution.
        # ==============================
        x = F.conv2d(
            input=x,
            weight=filter_bank.reshape(
                self.out_channels * self.group_order,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ),
            padding=(self.kernel_size - 1) // 2,
        )
        # ==============================

        # reshape output signal to shape [bz, #out, group order, h, w].
        # ==============================
        x = x.view(x.shape[0], self.out_channels, self.group_order, x.shape[-1], x.shape[-2])
        # ==============================
        if self.activation:
            return F.leaky_relu(x)
        return x
