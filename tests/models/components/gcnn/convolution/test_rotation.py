import pytest
import torch

from src.models.components.gcnn.convolution.rotation import GroupConvolution
from src.models.components.gcnn.convolution.rotation_rui import RuiGroupConvolution


@pytest.mark.skip
@pytest.mark.parametrize("group_order", [2, 4])
def test_gconv_forward(group_order: int):
    in_channels, out_channels, kernel_size = 3, 5, 3
    net1 = RuiGroupConvolution(in_channels, out_channels, kernel_size, group_order)
    net2 = GroupConvolution(in_channels, out_channels, kernel_size, group_order)
    net2.kernel = net1.kernel  # Copy the kernel
    B, H, W = 2, 5, 5
    x = torch.rand(B, in_channels, group_order, H, W)
    y1 = net1(x)
    y2 = net2(x)

    assert y1.shape == y2.shape
    assert torch.allclose(y1, y2)
