import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.components.gcnn.lifting.utils import generate_rot_filter_bank


class RLiftingConvCn(torch.nn.Module):
    """Relaxed lifting convolution Layer for 2D finite rotation group"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_order: int,
        num_filter_banks: int,
        activation: bool = True,
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param group_order: the order of 2d finite rotation group (e.g., 4 for C4 group)
        :param num_filter_banks: number of filter banks
        :param activation: whether to apply relu in the end
        """
        super().__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation
        self.padding = ((self.kernel_size - 1) // 2,)

        self.relaxed_weights = torch.nn.Parameter(
            torch.ones(num_filter_banks, group_order)
        )

        self.kernel = torch.nn.Parameter(
            torch.zeros(
                self.num_filter_banks,  # Additional dimension
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        torch.nn.init.kaiming_uniform_(self.kernel.data, a=np.sqrt(5))

    def generate_filter_bank(self):
        """Obtain a stack of rotated filters.
        :return: a tensor of shape [num_filter_banks, #out, group_order, #in, k, k]
        """
        # self.kernel: [num_filter_banks, #out, #in, k, k]
        weights = self.kernel.flatten(0, 1)
        # weights: [num_filter_banks * #out, #in, k, k]
        filter_bank = generate_rot_filter_bank(weights, self.group_order)
        # filter_bank: [num_filter_banks * #out, group_order, #in, k, k]
        filter_bank = filter_bank.unflatten(
            0, (self.num_filter_banks, self.out_channels)
        )
        # filter_bank: [num_filter_banks, #out, group_order, #in, k, k]
        return filter_bank

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor of shape [B, #in, H, W]
        :return: output tensor of shape [B, #out, group_order, H, W]
        """
        filter_bank = self.generate_filter_bank()
        # filter_bank: Tensor[num_filter_banks, #out, group_order, #in, k, k]

        # For each rotation, we have a linear combination of multiple filters with different coefficients
        relaxed_conv_weights = torch.sum(
            self.relaxed_weights.view(  # reshape for broadcast mult
                self.num_filter_banks, 1, self.group_order, 1, 1, 1
            )
            * filter_bank,
            dim=0,
        )
        x = F.conv2d(x, relaxed_conv_weights.flatten(0, 1), padding=self.padding)
        # x: Tensor[B, #out * group_order, H, W]
        x = x.unflatten(1, (self.out_channels, self.group_order))
        # x: Tensor[B, #out, group_order, H, W]
        if self.activation:
            return F.leaky_relu(x)
        return x
