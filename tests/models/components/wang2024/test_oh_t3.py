from gconv.gnn import GSeparableConvSE3, GLiftingConvSE3, GConvSE3, RGLiftingConvSE3
from gconv.geometry.groups import so3
from src.models.components.wang2024.oh_t3 import GCNNOhT3, RGCNNOhT3
import torch
import pytest



@pytest.mark.parametrize("classifier", [False, True])
def test_ohgcnn(classifier: bool):
    in_channels, out_channels, kernel_size, hidden_dim, num_gconvs = 3, 3, 3, 3, 3
    model = GCNNOhT3(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        num_gconvs=num_gconvs,
        classifier=classifier,
        sigmoid=True,
    )
    x = torch.randn(1, in_channels, 64, 64, 64)
    y = model(x)
    if classifier:
        assert y.shape == (1, out_channels)
    else:
        assert y.shape == (1, out_channels, 24, 64, 64, 64)


@pytest.mark.parametrize("classifier", [False, True])
def test_ohrgcnn(classifier: bool):
    in_channels, out_channels, kernel_size, hidden_dim, num_gconvs = 3, 3, 3, 3, 3
    num_filter_banks = 2
    model = RGCNNOhT3(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filter_banks=num_filter_banks,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        num_gconvs=num_gconvs,
        classifier=classifier,
        sigmoid=True,
    )
    x = torch.randn(1, in_channels, 64, 64, 64)
    y = model(x)
    if classifier:
        assert y.shape == (1, out_channels)
    else:
        assert y.shape == (1, out_channels, 24, 64, 64, 64)
