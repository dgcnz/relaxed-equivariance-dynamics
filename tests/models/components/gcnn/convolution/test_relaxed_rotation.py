import torch
import pytest
from src.models.components.gcnn.convolution.relaxed_rotation import RGroupConvCn
from src.models.components.gcnn.convolution.relaxed_rotation_rui import RuiRGroupConvCn


@pytest.mark.skip
def test_conv_einsum_equivalence():
    num_filter_banks, out_channels, group_order, in_channels, kernel_size = (
        2,
        6,
        4,
        3,
        5,
    )
    # filter_bank: Tensor[num_filter_banks, #out, group_order, #in, group_order, k, k]
    relaxed_weights = torch.randn((group_order, num_filter_banks))
    filter_bank = torch.randn(
        (
            num_filter_banks,
            out_channels,
            group_order,
            in_channels,
            group_order,
            kernel_size,
            kernel_size,
        )
    )

    # Run the einsum operation
    relaxed_conv_weights_einsum = torch.einsum(
        "na, aon... -> on...", relaxed_weights, filter_bank
    )
    # relaxed_conv_weights_einsum: Tensor[out_channels, group_order, in_channels, group_order, kernel_size, kernel_size]

    # Run equivalent operation with mult and sum
    relaxed_conv_weights_fast = torch.sum(
        relaxed_weights.transpose(0, 1).view(
            num_filter_banks, 1, group_order, 1, 1, 1, 1
        )
        * filter_bank,
        dim=0,
    )

    # Check that the results are the same
    assert torch.allclose(relaxed_conv_weights_einsum, relaxed_conv_weights_fast)

@pytest.mark.skip
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("group_order", [16])
@pytest.mark.parametrize("num_filter_banks", [64])
@pytest.mark.parametrize("in_channels, out_channels", [(16, 16)])
@pytest.mark.parametrize("kernel_size", [7])
@pytest.mark.parametrize("method", ["fast_gf", "fast_gl", "einsum"])
def test_benchmark_conv(
    benchmark,
    device: str,
    group_order: int,
    num_filter_banks: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    method: str,
):
    relaxed_weights = torch.randn((group_order, num_filter_banks), device=device)
    filter_bank = torch.randn(
        (
            num_filter_banks,
            out_channels,
            group_order,
            in_channels,
            group_order,
            kernel_size,
            kernel_size,
        ),
        device=device,
    )

    def fast():
        return torch.sum(
            relaxed_weights.transpose(0, 1).view(
                num_filter_banks, 1, group_order, 1, 1, 1, 1
            )
            * filter_bank,
            dim=0,
        )

    def fast_group_last():
        return torch.sum(
            relaxed_weights.view(num_filter_banks, 1, group_order, 1, 1, 1, 1)
            * filter_bank,
            dim=0,
        )

    def einsum():
        return torch.einsum("na, aon... -> on...", relaxed_weights, filter_bank)

    if method == "fast_gf":
        fun = fast
    elif method == "fast_gl":
        relaxed_weights = relaxed_weights.transpose(0, 1)
        fun = fast_group_last
    else:
        fun = einsum
    benchmark(fun)



@pytest.mark.skip
@pytest.mark.parametrize("group_order", [2, 4])
def test_relaxed_convolution_forward_even(group_order: int):
    num_filter_banks, in_channels, out_channels, kernel_size = 2, 3, 4, 5
    # Create the relaxed rotation group convolution layer
    conv1 = RGroupConvCn(
        num_filter_banks=num_filter_banks,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        group_order=group_order,
    )
    conv2 = RuiRGroupConvCn(
        num_filter_banks=num_filter_banks,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        group_order=group_order,
    )
    conv1.kernel = conv2.kernel
    conv1.relaxed_weights = conv2.relaxed_weights
    
    B, H, W = 2, 7, 7
    x = torch.randn((B, in_channels, group_order, H, W))
    y1 = conv1(x)
    y2 = conv2(x)

    assert y1.shape == y2.shape
    assert torch.allclose(y1, y2)