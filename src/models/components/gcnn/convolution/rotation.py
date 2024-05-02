import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TTF


class GroupConvolution(nn.Module):
    """Group Convolution Layer for finite rotation group"""

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
        self.padding = (self.kernel_size - 1) // 2

        self.kernel = torch.nn.Parameter(
            torch.zeros(
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
        """Generate a stack of rotated and cyclic shifted filters
        :param kernel: the kernel tensor of shape [#out, #in, group_order, k, k]
        :return: a tensor of shape [#out, group_order, #in, group_order, k, k]
        """
        filter_bank = []
        out_channels, in_channels, group_order, k, _ = kernel.shape
        filter = kernel.flatten(0, 1)
        # filter: Tensor[#out * #in, group_order, k, k]

        # TODO: is it better to vectorize this operation or not?
        for i in range(group_order):
            rot_filter = TTF.rotate(filter, -360 / group_order * i)
            shifted_rot_filter = torch.roll(rot_filter, shifts=i, dims=1)
            shifted_rot_filter = shifted_rot_filter.unflatten(
                0, (out_channels, in_channels)
            )
            filter_bank.append(shifted_rot_filter)
        filter_bank = torch.stack(filter_bank, dim=1)
        return filter_bank

    def forward(self, x: torch.Tensor):
        """Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, group_order, H, W]
        :return: output tensor of shape [B, #out, group_order, H, W]
        """

        filter_bank = GroupConvolution.generate_filter_bank(self.kernel)
        # filter_bank: Tensor[#out, group_order, #in, group_order, H, W]

        x = torch.nn.functional.conv2d(
            input=x.flatten(1, 2),  # [B, #in * group_order, H, W]
            weight=filter_bank.reshape(
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
