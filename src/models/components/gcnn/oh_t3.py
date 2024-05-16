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
        self.group_kernel_size = 48
        self.grid_Oh = so3.quat_to_matrix(so3.octahedral())

        self.gconvs = nn.Sequential(
            GLiftingConvSE3(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                groups=1,  # TODO: how to use separable convs?
                padding=kernel_size // 2,
                group_kernel_size=self.group_kernel_size,
                grid_H=self.grid_Oh,
                permute_output_grid=False,
                maks=False,
            ),
            *[
                GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    group_kernel_size=self.group_kernel_size,
                    groups=1,  # TODO: how to use separable convs?
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,
                )
                for _ in range(num_gconvs - 2)
            ],
            GSeparableConvSE3(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                group_kernel_size=self.group_kernel_size,
                groups=1,  # TODO: how to use separable convs?
                padding=kernel_size // 2,
                permute_output_grid=False,
                grid_H=self.grid_Oh,
                mask=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, H, W]
        """
        out = self.gconvs(x)
        # out: [N, #out, group_order, H, W]
        # out = torch.mean(out, dim=2)
        # out: [N, #out, H, W]

        # if self.classifier:
        #     out = out.mean((2, 3))
        # if self.sigmoid:
        #     out = out.sigmoid()
        return out
