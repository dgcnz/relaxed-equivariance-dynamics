import pytest
import torch
import math
from src.models.components.gcnn.lifting.rotation import LiftingConvCn
from src.models.components.gcnn.lifting.rotation_rui import RuiLiftingConvCn

import gconv.gnn as gnn
from src.models.components.gcnn.lifting.rotation_rui import RuiLiftingConvCn


@pytest.mark.skip
@pytest.mark.parametrize("group_order", [2, 4])
def test_lifting_forward_even(group_order: int):
    in_channels, out_channels, kernel_size = 3, 5, 3
    net1 = RuiLiftingConvCn(in_channels, out_channels, kernel_size, group_order)
    net2 = LiftingConvCn(in_channels, out_channels, kernel_size, group_order)
    net2.kernel = net1.kernel  # Copy the kernel
    B, H, W = 2, 5, 5
    x = torch.rand(B, in_channels, H, W)

    y1 = net1(x)
    y2 = net2(x)

    assert y1.shape == y2.shape
    assert torch.allclose(y1, y2)


@pytest.mark.skip
@pytest.mark.parametrize("group_order", [5])
def test_lifting_forward_odd(group_order: int):
    in_channels, out_channels, kernel_size = 3, 5, 3
    net1 = RuiLiftingConvCn(in_channels, out_channels, kernel_size, group_order)
    net2 = LiftingConvCn(in_channels, out_channels, kernel_size, group_order)
    net2.kernel = net1.kernel  # Copy the kernel
    B, H, W = 2, 5, 5
    x = torch.rand(B, in_channels, H, W)

    y1 = net1(x)
    y2 = net2(x)

    assert y1.shape == y2.shape
    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("group_order", [2, 4])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_gconv_lifting_forward_even(
    kernel_size: int, group_order: int, out_channels: int, in_channels: int
):
    x = torch.randn(1, in_channels, 28, 28)
    torch.manual_seed(0)
    # For some reason the kernel is rotated in the opposite direction
    rui_grid_H = torch.tensor([math.pi * 2 / group_order * ((group_order - i)%group_order)  for i in range(group_order)]).unsqueeze(-1)
    lift1 = RuiLiftingConvCn(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        group_order=group_order,
        activation=False,
    )
    lift2 = gnn.GLiftingConvSE2(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        group_kernel_size=group_order,
        padding=kernel_size // 2,
        sampling_mode="nearest",
        permute_output_grid=False,
        mask=False,
        grid_H=rui_grid_H,
    )
    # For some reason gnn transposes the kernel, not really important
    lift1.kernel = torch.nn.Parameter(lift2.kernel.weight.transpose(-1, -2))

    y1 = lift1(x)
    y2, _ = lift2(x)
    assert torch.allclose(y1, y2)
