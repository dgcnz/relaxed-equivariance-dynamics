from gconv.gnn import GSeparableConvSE3, GLiftingConvSE3
from gconv.geometry.groups import so3
import torch
from torch import Tensor
import torch.nn as nn


class GCNNOhT3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,
        num_gconvs: int,
        classifier: bool = False,
        sigmoid: bool = False,
    ):
        assert num_gconvs >= 2
        super().__init__()

        self.classifier = classifier
        self.sigmoid = sigmoid
        self.grid_Oh = so3.quat_to_matrix(so3.octahedral())
        self.dims = [in_channels] + [hidden_dim] * (num_gconvs - 2) + [out_channels]
        self.lift = GLiftingConvSE3(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                groups=1,  # TODO: how many groups should we use?
                padding=kernel_size // 2,
                grid_H=self.grid_Oh,
                permute_output_grid=False,
                mask=False,
            )
        self.gconvs = nn.ModuleList([
                GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=self.dims[i],
                    kernel_size=kernel_size,
                    groups=1,  # TODO: how many groups should we use?
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,
                )
                for i in range(1, num_gconvs)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, H, W]
        """
        z, H = self.lift(x, self.grid_Oh)
        for gconv in self.gconvs:
            z, H = gconv(z, H)
        if self.classifier:
            z = z.mean((2, 3, 4, 5))
        if self.sigmoid:
            z = z.sigmoid()
        return z
