import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.image_utils import rot_img


class RuiGroupConvolution(nn.Module):
    """Group Convolution Layer for finite rotation group.

    Attributes:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size
        group_order: the order of rotation groups
        activation: whether to use relu.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_order,
        activation=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation

        # Initialize an unconstrained kernel.
        # the weights have an additional group order dimension.
        self.kernel = torch.nn.Parameter(
            torch.zeros(
                self.out_channels,
                self.in_channels,
                self.group_order,  # this is different from the lifting convolution
                self.kernel_size,
                self.kernel_size,
            )
        )

        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))

    def generate_filter_bank(self):
        # Obtain a stack of rotated and cyclic shifted filters
        filter_bank = []
        filter = self.kernel.reshape(
            self.out_channels * self.in_channels,
            self.group_order,
            self.kernel_size,
            self.kernel_size,
        )

        for i in range(self.group_order):
            # planar rotation
            rotated_filter = rot_img(filter, -np.pi * 2 / self.group_order * i)

            # cyclic shift
            shifted_indices = torch.roll(torch.arange(0, self.group_order, 1), shifts=i)
            shifted_rotated_filter = rotated_filter[:, shifted_indices]

            filter_bank.append(
                shifted_rotated_filter.reshape(
                    self.out_channels,
                    self.in_channels,
                    self.group_order,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        # reshape output signal to shape [#out, g_order, #in, g_order, k, k].
        filter_bank = torch.stack(filter_bank).transpose(0, 1)
        return filter_bank

    def forward(self, x: torch.Tensor):
        # input shape: [bz, in, group order, x, y]
        # output shape: [bz, out, group order, x, y]

        # Generate filter bank with shape [#out, g_order, #in, g_order, h, w]
        filter_bank = self.generate_filter_bank()

        # Reshape filter_bank to use F.conv2d
        # [#out, g_order, #in, g_order, h, w] -> [#out*g_order, #in*g_order, h, w]
        # ==============================
        x = torch.nn.functional.conv2d(
            input=x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]),
            weight=filter_bank.reshape(
                self.out_channels * self.group_order,
                self.in_channels * self.group_order,
                self.kernel_size,
                self.kernel_size,
            ),
            padding=(self.kernel_size - 1) // 2,
        )

        # Reshape signal back [bz, #out * g_order, h, w] -> [bz, out, g_order, h, w]
        x = x.view(x.shape[0], self.out_channels, self.group_order, x.shape[-2], x.shape[-1])
        # ========================

        if self.activation:
            return F.leaky_relu(x)
        return x
