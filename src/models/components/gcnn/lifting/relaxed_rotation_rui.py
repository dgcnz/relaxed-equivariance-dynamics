import math

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.image_utils import rot_img


class RuiCNRelaxedLiftingConvolution(torch.nn.Module):
    """Relaxed lifting convolution Layer for 2D finite rotation group"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_order,  # the order of 2d finite rotation group
        num_filter_banks,
        activation=True,  # whether to apply relu in the end
    ):
        super().__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation

        # The relaxed weights are initialized as equal
        # they do not need to be equal across different filter bank
        self.relaxed_weights = torch.nn.Parameter(
            torch.ones(num_filter_banks, group_order).float()
        )

        # Initialize an unconstrained kernel.
        self.kernel = torch.nn.Parameter(
            torch.zeros(
                self.num_filter_banks,  # Additional dimension
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))

    def generate_filter_bank(self):
        """Obtain a stack of rotated filters"""
        weights = self.kernel.reshape(
            self.num_filter_banks * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        filter_bank = torch.stack(
            [
                rot_img(weights, -np.pi * 2 / self.group_order * i)
                for i in range(self.group_order)
            ]
        )
        filter_bank = filter_bank.transpose(0, 1).reshape(
            self.num_filter_banks,  # Additional dimension
            self.out_channels,
            self.group_order,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        return filter_bank

    def forward(self, x):
        # input shape: [bz, #in, h, w]
        # output shape: [bz, #out, group order, h, w]

        # generate filter bank given input group order
        filter_bank = self.generate_filter_bank()

        # for each rotation, we have a linear combination of multiple filters with different coefficients.
        relaxed_conv_weights = torch.einsum(
            "na, noa... -> oa...", self.relaxed_weights, filter_bank
        )

        # concatenate the first two dims before convolution.
        # ==============================
        x = F.conv2d(
            input=x,
            weight=relaxed_conv_weights.reshape(
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
        x = x.view(
            x.shape[0], self.out_channels, self.group_order, x.shape[-1], x.shape[-2]
        )
        # ==============================

        if self.activation:
            return F.leaky_relu(x)
        return x
