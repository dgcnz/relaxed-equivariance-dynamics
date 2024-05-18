from gconv.gnn import GSeparableConvSE3, GLiftingConvSE3, GConvSE3, RGLiftingConvSE3
from gconv.geometry.groups import so3
from src.models.components.wang2024.oh_t3 import GCNNOhT3
import torch


def test_oh_separable_conv():
    H0 = so3.quat_to_matrix(so3.octahedral())
    in_channels, hidden_dim, kernel_size = 3, 3, 3

    lift = GLiftingConvSE3(
        in_channels=in_channels,
        out_channels=hidden_dim,
        kernel_size=kernel_size,
        groups=1,
        padding=kernel_size // 2,
        grid_H=H0,
        permute_output_grid=False,
        mask=False,
    )
    gsconv = GSeparableConvSE3(
        in_channels=hidden_dim,
        out_channels=hidden_dim,
        kernel_size=kernel_size,
        groups=1,
        padding=kernel_size // 2,
        permute_output_grid=False,
        grid_H=H0,
        mask=False,
    )
    gconv = GConvSE3(
        in_channels=hidden_dim,
        out_channels=hidden_dim,
        kernel_size=kernel_size,
        groups=1,
        padding=kernel_size // 2,
        permute_output_grid=False,
        grid_H=H0,
        mask=False,
    )
    x = torch.randn(1, in_channels, 48, 48, 48)
    z, H1 = lift(x)
    ws, H2s = gsconv(z, H1) 
    w, H2 = gconv(z, H1) 
    print(ws.shape, w.shape)

def test_oh_lifting_rconv():
    H0 = so3.quat_to_matrix(so3.octahedral())
    in_channels, hidden_dim, kernel_size, num_filter_banks = 3, 3, 3, 6
    lift = RGLiftingConvSE3(
        in_channels=in_channels,
        out_channels=hidden_dim,
        num_filter_banks=num_filter_banks,
        kernel_size=kernel_size,
        groups=1,
        padding=kernel_size // 2,
        grid_H=H0,
        permute_output_grid=False,
        mask=False,
    )

    x = torch.randn(1, in_channels, 48, 48, 48)
    z, H1 = lift(x)


def test_ohgcnn():
    in_channels, out_channels, kernel_size, hidden_dim, num_gconvs = 3, 3, 3, 3, 3
    model = GCNNOhT3(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        num_gconvs=num_gconvs,
        classifier=True,
        sigmoid=True,
    )
    x = torch.randn(1, in_channels, 48, 48, 48)
    y = model(x)
    assert y.shape == (1, out_channels)