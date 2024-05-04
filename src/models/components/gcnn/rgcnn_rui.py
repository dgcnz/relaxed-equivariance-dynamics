from src.models.components.gcnn.convolution.relaxed_rotation_rui import (
    RuiRGroupConvCn,
)
from src.models.components.gcnn.lifting.relaxed_rotation_rui import (
    RuiRLiftingConvCn,
)
import torch


class RuiCnRGCNN(torch.nn.Module):
    """Rui's implementation of C(n) equivariant relaxed group convolutional neural network"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        hidden_dim,
        group_order,  # the order of 2d finite rotation group
        num_gconvs,  # number of group conv layers
        num_filter_banks,
        classifier=False,
        sigmoid=False,
    ):
        super().__init__()

        self.gconvs = []
        self.classifier = classifier
        self.sigmoid = sigmoid

        self.gconvs = [
            RuiRLiftingConvCn(
                in_channels,
                hidden_dim,
                kernel_size,
                group_order,
                num_filter_banks,
                True,
            )
        ]

        for i in range(num_gconvs - 2):
            self.gconvs.append(
                RuiRGroupConvCn(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    group_order,
                    num_filter_banks,
                    True,
                )
            )

        self.gconvs.append(
            RuiRGroupConvCn(
                hidden_dim,
                out_channels,
                kernel_size,
                group_order,
                num_filter_banks,
                False,
            )
        )

        self.gconvs = torch.nn.Sequential(*self.gconvs)

    def forward(self, x):
        # average over h axis or not
        out = self.gconvs(x).mean(2)

        if self.classifier:
            out = out.mean((2, 3))
        if self.sigmoid:
            out = out.sigmoid()
        return out
