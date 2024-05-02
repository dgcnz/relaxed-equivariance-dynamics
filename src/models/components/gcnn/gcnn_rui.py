from src.models.components.gcnn.lifting.rotation_rui import RuiCNLiftingConvolution
from src.models.components.gcnn.convolution.rotation_rui import RuiGroupConvolution
import torch.nn as nn
import torch


class RuiGroupEquivariantCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,
        group_order: int,
        num_gconvs: int,  # number of group convolution layers.
        classifer: bool = False,
        sigmoid: bool = False,
    ):
        super().__init__()

        self.gconvs = []
        # First Layer
        self.gconvs.append(
            RuiCNLiftingConvolution(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                group_order=group_order,
                activation=True,
            )
        )
        # Middle Layers

        for i in range(num_gconvs - 2):
            self.gconvs.append(
                RuiGroupConvolution(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    group_order=group_order,
                    activation=True,
                )
            )

        # Final Layer # To generate equivariant outputs
        self.gconvs.append(
            RuiGroupConvolution(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                group_order=group_order,
                activation=False,
            )
        )
        self.gconvs = nn.Sequential(*self.gconvs)
        self.classifer = classifer
        self.sigmoid = sigmoid

    def forward(self, x):
        # x [bz, c_in, h, w]
        out = self.gconvs(x)

        # functions on (g,x,y) -> functions on (x,y)
        # [bz, c_out, |G|, H, W] -> [bz, c_out, H, W]
        out = torch.mean(out, dim=2)

        # If we want to have a invariant classifer, we can average over the spatial dimensions.
        if self.classifer:
            out = torch.mean(out, dim=(2, 3))
        if self.sigmoid:
            out = out.sigmoid()
        return out
