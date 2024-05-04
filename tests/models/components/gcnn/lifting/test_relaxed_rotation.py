import torch
import pytest
from src.models.components.gcnn.lifting.relaxed_rotation import CNRelaxedLiftingConvolution
from src.models.components.gcnn.lifting.relaxed_rotation_rui import RuiCNRelaxedLiftingConvolution

@pytest.mark.skip
@pytest.mark.parametrize("group_order", [2, 4])
def test_relaxed_lifting_forward_even(group_order: int):
    in_channels, out_channels, kernel_size, num_filter_banks = 3, 5, 3, 2
    net1 = RuiCNRelaxedLiftingConvolution(
        in_channels, out_channels, kernel_size, group_order, num_filter_banks
    )
    net2 = CNRelaxedLiftingConvolution(
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
def test_einsum_equivalence():
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
