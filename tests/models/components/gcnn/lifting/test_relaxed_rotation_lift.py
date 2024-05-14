import torch
import pytest
from src.models.components.gcnn.lifting.relaxed_rotation import RLiftingConvCn
from src.models.components.gcnn.lifting.relaxed_rotation_rui import RuiRLiftingConvCn
import math
from gconv import gnn


@pytest.mark.skip
@pytest.mark.parametrize("group_order", [2, 4])
def test_relaxed_lifting_forward_even(group_order: int):
    in_channels, out_channels, kernel_size, num_filter_banks = 3, 5, 3, 2
    net1 = RuiRLiftingConvCn(
        in_channels, out_channels, kernel_size, group_order, num_filter_banks
    )
    net2 = RLiftingConvCn(
        in_channels, out_channels, kernel_size, group_order, num_filter_banks
    )
    net2.kernel = net1.kernel
    net2.relaxed_weights = net1.relaxed_weights
    B, H, W = 2, 5, 5
    x = torch.rand(B, in_channels, H, W)

    y1 = net1(x)
    y2 = net2(x)

    assert y1.shape == y2.shape
    assert torch.allclose(y1, y2)


@pytest.mark.skip
def test_einsum_equivalence_lifting():
    # Create mock tensors
    num_filter_banks, out_channels, group_order, in_channels, kernel_size = (
        2,
        6,
        4,
        3,
        5,
    )
    relaxed_weights = torch.randn((num_filter_banks, group_order))
    filter_bank = torch.randn(
        (
            num_filter_banks,
            out_channels,
            group_order,
            in_channels,
            kernel_size,
            kernel_size,
        )
    )

    def einsum():
        return torch.einsum("na, noa... -> oa...", relaxed_weights, filter_bank)

    def fast():
        return torch.sum(
            relaxed_weights.view(num_filter_banks, 1, group_order, 1, 1, 1)
            * filter_bank,
            dim=0,
        )

    # Run the einsum operation
    result_einsum = einsum()
    # result_einsum: [out_channels, group_order, in_channels, kernel_size, kernel_size]
    result_fast = fast()

    # Check that the two results are close
    assert torch.allclose(result_einsum, result_fast, atol=1e-6)


@pytest.mark.skip
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("group_order", [16])
@pytest.mark.parametrize("num_filter_banks", [64])
@pytest.mark.parametrize("in_channels, out_channels", [(16, 16)])
@pytest.mark.parametrize("kernel_size", [7])
def test_benchmark_einsum(
    benchmark,
    device: str,
    group_order: int,
    num_filter_banks: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
):
    # Create mock tensors
    relaxed_weights = torch.randn((num_filter_banks, group_order), device=device)
    filter_bank = torch.randn(
        (
            num_filter_banks,
            out_channels,
            group_order,
            in_channels,
            kernel_size,
            kernel_size,
        ),
        device=device,
    )

    def einsum():
        return torch.einsum("na, noa... -> oa...", relaxed_weights, filter_bank)

    benchmark(einsum)


@pytest.mark.skip
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("group_order", [16])
@pytest.mark.parametrize("num_filter_banks", [64])
@pytest.mark.parametrize("in_channels, out_channels", [(16, 16)])
@pytest.mark.parametrize("kernel_size", [7])
def test_benchmark_fast(
    benchmark,
    device: str,
    group_order: int,
    num_filter_banks: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
):
    # Create mock tensors
    num_filter_banks, out_channels, group_order, in_channels, kernel_size = (
        2,
        6,
        4,
        3,
        5,
    )
    relaxed_weights = torch.randn((num_filter_banks, group_order), device=device)
    filter_bank = torch.randn(
        (
            num_filter_banks,
            out_channels,
            group_order,
            in_channels,
            kernel_size,
            kernel_size,
        ),
        device=device,
    )

    def fast():
        return torch.sum(
            relaxed_weights.view(num_filter_banks, 1, group_order, 1, 1, 1)
            * filter_bank,
            dim=0,
        )

    # Run the einsum operation
    # result_einsum: [out_channels, group_order, in_channels, kernel_size, kernel_size]
    benchmark(fast)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [1, 3])
@pytest.mark.parametrize("group_order", [2, 4])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("num_filter_banks", [1, 2])
def test_gconv_lifting_forward_even(
    kernel_size: int,
    group_order: int,
    out_channels: int,
    in_channels: int,
    num_filter_banks: int,
):
    x = torch.randn(1, in_channels, 28, 28)
    torch.manual_seed(0)
    # For some reason the kernel is rotated in the opposite direction
    rui_grid_H = torch.tensor(
        [
            math.pi * 2 / group_order * ((group_order - i) % group_order)
            for i in range(group_order)
        ]
    ).unsqueeze(-1)
    num_filter_banks = 2
    lift1 = RuiRLiftingConvCn(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        group_order=group_order,
        num_filter_banks=num_filter_banks,
        activation=False,
    )
    lift2 = gnn.RGLiftingConvSE2(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filter_banks=num_filter_banks,
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
