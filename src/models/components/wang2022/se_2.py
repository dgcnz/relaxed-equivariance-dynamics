from gconv.gnn import GConvSE2, GLiftingConvSE2
from torch import nn, Tensor
import torch

class GCNNSE2(nn.Module):
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
        self.dims = [in_channels] + [hidden_dim] * (num_gconvs - 2) + [out_channels]
        self.lift = GLiftingConvSE2(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            permute_output_grid=False,
            mask=False,
        )
        self.gconvs = nn.ModuleList(
            [
                GConvSE2(
                    in_channels=hidden_dim,
                    out_channels=self.dims[i],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    mask=False,
                    group_sampling_mode="nearest"
                )
                for i in range(1, num_gconvs)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, H, W]
        """
        z, H = self.lift(x)
        # z: [N, #out, group_order, H, W]
        for gconv in self.gconvs:
            z, H = gconv(z, H)
        z = torch.mean(z, dim=2)
        # z: [N, #out, H, W]
        if self.classifier:
            z = z.mean((2, 3))
        if self.sigmoid:
            z = z.sigmoid()
        return z
    
