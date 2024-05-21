import pytest
import torch

from src.models.components.gcnn.gcnn import CnGCNN
from src.models.components.gcnn.gcnn_rui import RuiCnGCNN


# @pytest.mark.skip
@pytest.mark.parametrize("group_order", [2, 4])
def test_gcnn_even(group_order: int):
    in_channels, out_channels, kernel_size, hidden_dim, num_gconvs = 3, 5, 3, 10, 3
    B, H, W = 2, 6, 6
    x = torch.rand(B, in_channels, H, W)
    net1 = CnGCNN(
        in_channels, out_channels, kernel_size, hidden_dim, group_order, num_gconvs
    )
    net2 = RuiCnGCNN(
        in_channels, out_channels, kernel_size, hidden_dim, group_order, num_gconvs
    )
    for i in range(num_gconvs):
        net2.gconvs[i].kernel = net1.gconvs[i].kernel

    y1 = net1(x)
    y2 = net2(x)

    assert y1.shape == y2.shape
    assert torch.allclose(y1, y2)
