import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.gcnn.convolution.rotation import GroupConvolution
from src.models.components.gcnn.lifting.rotation import CNLiftingConvolution


class GroupEquivariantCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,
        group_order: int,
        num_gconvs: int,
        classifer: bool = False,
        sigmoid: bool = False,
    ):
        """
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size of the convolutional layers.
        :param hidden_dim: Number of hidden dimensions.
        :param group_order: Order of the group.
        :param num_gconvs: Number of group convolution layers. Must be at least 2 (counting lifting conv).
        :param classifer: If True, the output is averaged over the spatial dimensions.
        :param sigmoid: If True, the output is passed through a sigmoid function.
        """
        assert num_gconvs >= 2
        super().__init__()
        self.gconvs = nn.Sequential(
            CNLiftingConvolution(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                group_order=group_order,
                activation=True,
            ),
            *[
                GroupConvolution(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    group_order=group_order,
                    activation=True,
                )
                for _ in range(num_gconvs - 2)
            ],
            GroupConvolution(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                group_order=group_order,
                activation=False,
            ),
        )
        self.classifer = classifer
        self.sigmoid = sigmoid

    def forward(self, x: Tensor) -> Tensor:
        out = self.gconvs(x)
        # out: [N, #out, group_order, H, W]
        out = torch.mean(out, dim=2)
        # out: [N, #out, H, W]

        # If we want to have a invariant classifer,
        # we can average over the spatial dimensions.
        if self.classifer:
            out = torch.mean(out, dim=(2, 3))
        if self.sigmoid:
            out = out.sigmoid()
        return out
