from src.models.components.wang2024.gcnn_oh_t3 import GCNNOhT3
from src.models.components.wang2024.rgcnn_oh_t3 import RGCNNOhT3
import torch


def test_ohgcnn():
    in_channels, out_channels, kernel_size, hidden_dim = 9, 3, 3, 2
    model = GCNNOhT3(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
    )
    x = torch.randn(1, in_channels, 2, 2, 2)
    y = model(x)
    assert y.shape == (1, out_channels, 8, 8, 8)


def test_ohrgcnn():
    in_channels, out_channels, kernel_size, hidden_dim = 9, 3, 3, 2
    num_filter_banks = 1
    model = RGCNNOhT3(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filter_banks=num_filter_banks,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
    )
    x = torch.randn(1, in_channels, 2, 2, 2)
    y = model(x)
    assert y.shape == (1, out_channels, 8, 8, 8)
