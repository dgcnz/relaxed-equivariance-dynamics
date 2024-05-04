import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TTF
from torch import Tensor


class RGroupConvCn(torch.nn.Module):
    """Relaxed group convolution Layer for 2D finite rotation group"""

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
        self.padding = (self.kernel_size - 1) // 2

        self.relaxed_weights = torch.nn.Parameter(
            torch.ones(group_order, num_filter_banks)
        )
        self.kernel = torch.nn.Parameter(
            torch.randn(
                self.num_filter_banks,  # additional dimension
                self.out_channels,
                self.in_channels,
                self.group_order,
                self.kernel_size,
                self.kernel_size,
            )
        )

        torch.nn.init.kaiming_uniform_(self.kernel.data, a=np.sqrt(5))

    @classmethod
    def generate_filter_bank(cls, kernel: torch.Tensor) -> torch.Tensor:
        """Obtain a stack of rotated and cyclic shifted filters

        :param kernel: the kernel tensor of shape [num_filter_banks, #out, #in, group_order, k, k]
        :return: a tensor of shape [num_filter_banks, #out, group_order, #in, group_order, k, k]
        """
        filter_bank = []
        num_filter_banks, out_channels, in_channels, group_order, *_ = kernel.shape
        weights = kernel.flatten(0, 2)
        # weights: Tensor[num_filter_banks * #out * #in, group_order, k, k]

        for i in range(group_order):
            rot_filter = TTF.rotate(weights, -360 / group_order * i)
            shifted_rot_filter = torch.roll(rot_filter, shifts=i, dims=1)
            shifted_rot_filter = shifted_rot_filter.unflatten(
                0, (num_filter_banks, out_channels, in_channels)
            )
            filter_bank.append(shifted_rot_filter)
        # filter_bank: Tensor[group_order, num_filter_banks, #out, #in, group_order, k, k]
        filter_bank = torch.stack(filter_bank, dim=2)
        # filter_bank: Tensor[num_filter_banks, #out, group_order, #in, group_order k, k]
        return filter_bank

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, group_order, H, W]
        :return: output tensor of shape [B, #out, group_order, H, W]
        """

        filter_bank = RGroupConvCn.generate_filter_bank(self.kernel)
        # filter_bank: Tensor[num_filter_banks, #out, group_order, #in, group_order, k, k]

        relaxed_conv_weights = torch.sum(
            self.relaxed_weights.transpose(0, 1).view(
                self.num_filter_banks, 1, self.group_order, 1, 1, 1, 1
            )
            * filter_bank,
            dim=0,
        )
        # relaxed_conv_weights: Tensor[#out, group_order, #in, group_order, k, k]

        x = torch.nn.functional.conv2d(
            input=x.flatten(1, 2),  # [B, #in * group_order, H, W]
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels * self.group_order,
                self.kernel_size,
                self.kernel_size,
            ),
            padding=self.padding,
        )
        x = x.unflatten(1, (self.out_channels, self.group_order))
        if self.activation:
            return F.leaky_relu(x)
        return x
