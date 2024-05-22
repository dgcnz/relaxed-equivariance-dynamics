from src.models.components.wang2022.se_2 import GCNNSE2
import torch
import pytest

@pytest.mark.parametrize("classifier", [True, False])
def test_gcnn_se2_forward(classifier: bool):
    in_channels = 3
    out_channels = 10
    kernel_size = 3
    hidden_dim = 64
    num_gconvs = 3
    sigmoid = False

    net = GCNNSE2(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        num_gconvs=num_gconvs,
        classifier=classifier,
        sigmoid=sigmoid,
    )

    x = torch.rand(2, in_channels, 32, 32)
    out = net(x)
    if classifier:
        assert out.shape == (2, out_channels)
    else:
        assert out.shape == (2, out_channels, 32, 32)