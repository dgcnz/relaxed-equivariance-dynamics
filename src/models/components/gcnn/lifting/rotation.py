import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TTF
import numpy as np


def generate_rot_filter_bank(kernel: Tensor, group_order: int) -> Tensor:
    """
    Generate a stack of rotated filters
    :param kernel: the kernel tensor of shape [#out, #in, k, k]
    :param group_order: the order of the rotation group (4 = C4)
    :return: a tensor of shape [#out, group_order, #in, k, k]
    """
    thetas = np.arange(group_order) * (-360 / group_order)
    return torch.stack([TTF.rotate(kernel, theta) for theta in thetas], dim=1)


class CNLiftingConvolution(nn.Module):
    """Lifting Convolution Layer for finite rotation group"""

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
        :param x: input tensor of shape [B, Cin, H, W]
        :return: output tensor of shape [B, Cout, group_order, H, W]
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
