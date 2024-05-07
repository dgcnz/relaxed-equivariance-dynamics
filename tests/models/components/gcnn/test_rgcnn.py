import pytest
import torch

from src.models.components.gcnn.rgcnn import CnRGCNN
from src.models.components.gcnn.rgcnn_rui import RuiCnRGCNN


@pytest.mark.skip
@pytest.mark.parametrize("group_order", [2, 4])
def test_rgcnn_even(group_order: int):
    in_channels, out_channels, kernel_size, hidden_dim, num_gconvs, num_filter_banks = (
        3,
        5,
        3,
        10,
        3,
        4,
    )
    B, H, W = 2, 5, 5
    x = torch.rand(B, in_channels, H, W)
    net1 = CnRGCNN(
        in_channels,
        out_channels,
        kernel_size,
        hidden_dim,
        group_order,
        num_gconvs,
        num_filter_banks,
    )
    net2 = RuiCnRGCNN(
        in_channels,
        out_channels,
        kernel_size,
        hidden_dim,
        group_order,
        num_gconvs,
        num_filter_banks,
    )
    for i in range(num_gconvs):
        net2.gconvs[i].kernel = net1.gconvs[i].kernel
        net2.gconvs[i].relaxed_weights = net1.gconvs[i].relaxed_weights

    y1 = net1(x)
    y2 = net2(x)

    assert y1.shape == y2.shape
    assert torch.allclose(y1, y2)
