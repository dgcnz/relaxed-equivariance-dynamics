from src.models.components.gcnn.convolution.relaxed_rotation import (
    RGroupConvCn,
)
from src.models.components.gcnn.lifting.relaxed_rotation import (
    RLiftingConvCn,
)
import torch
from torch import Tensor
import torch.nn as nn


class CnRGCNN(nn.Module):
    """C(n) equivariant relaxed group convolutional neural network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,
        group_order: int,
        num_gconvs: int,
        num_filter_banks: int,
        classifier: bool = False,
        sigmoid: bool = False,
    ):
        """
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size of the convolutional layers.
        :param hidden_dim: Number of hidden dimensions.
        :param group_order: Order of the group.
        :param num_gconvs: Number of group convolution layers. Must be at least 2 (counting lifting conv).
        :param num_filter_banks: Number of filter banks.
        :param classifier: If True, the output is averaged over the spatial dimensions.
        :param sigmoid: If True, the output is passed through a sigmoid function.
        """
        assert num_gconvs >= 2
        super().__init__()

        self.classifier = classifier
        self.sigmoid = sigmoid

        self.gconvs = nn.Sequential(
            RLiftingConvCn(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                group_order=group_order,
                num_filter_banks=num_filter_banks,
                activation=True,
            ),
            *[
                RGroupConvCn(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    group_order=group_order,
                    num_filter_banks=num_filter_banks,
                    activation=True,
                )
                for _ in range(num_gconvs - 2)
            ],
            RGroupConvCn(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                group_order=group_order,
                num_filter_banks=num_filter_banks,
                activation=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, H, W]
        """
        out = self.gconvs(x)
        # out: [N, #out, group_order, H, W]
        out = torch.mean(out, dim=2)
        # out: [N, #out, H, W]

        if self.classifier:
            out = out.mean((2, 3))
        if self.sigmoid:
            out = out.sigmoid()
        return out
