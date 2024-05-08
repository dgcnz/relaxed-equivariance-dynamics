from torch import nn
from torch import Tensor
from src.models.components.gcnn.gcnn import CnGCNN


class GCNNClassifier(nn.Module):
    """A toy classifier to test GCNN"""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        hidden_dim: int,
        group_order: int,
        num_gconvs: int,
        num_classes: int,
    ):
        super().__init__()
        self.gcnn = CnGCNN(
                in_channels,
                num_classes,
                kernel_size,
                hidden_dim,
                group_order,
                num_gconvs,
                classifer=True,
                sigmoid=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.gcnn(x)
        return x
