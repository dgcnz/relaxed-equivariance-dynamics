import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from src.models.components.gcnn.lifting.utils import generate_rot_filter_bank


class CNLiftingConvolution(nn.Module):
    """Lifting Convolution Layer for finite rotation group."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_order: int,
        activation: bool = True,
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param group_order: the order of the rotation group (4 = C4)
        :param activation: whether to use relu
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation

        self.kernel = nn.Parameter(
            torch.zeros(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            )
        )
        nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor of shape [B, #in, H, W]
        :return: output tensor of shape [B, #out, group_order, H, W]
        """
        filter_bank = generate_rot_filter_bank(self.kernel, self.group_order)
        # filter_bank: Tensor[#out, group_order, #in, k, k]
        x = F.conv2d(x, filter_bank.flatten(0, 1), padding=(self.kernel_size - 1) // 2)
        # x: Tensor[B, #out * group_order, H, W]
        x = x.unflatten(1, (self.out_channels, self.group_order))
        # x: Tensor[B, #out, group_order, H, W]
        if self.activation:
            # TODO: should the activation be decoupled from this block?
            return F.leaky_relu(x)
        return x
