import e2cnn
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from e2cnn.nn.modules.r2_conv.basisexpansion_singleblock import block_basisexpansion
from e2cnn.nn.modules.r2_conv.r2convolution import compute_basis_params


class Relaxed_Rot_SteerConv(torch.nn.Module):
    def __init__(
        self,
        in_frames: int,
        out_frames: int,
        kernel_size: int,
        N: int,
        first_layer: bool = False,
        last_layer: bool = False,
    ):
        super().__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N=N)
        self.last_layer = last_layer
        self.first_layer = first_layer
        self.kernel_size = kernel_size

        if self.first_layer:
            self.feat_type_in = e2cnn.nn.FieldType(
                r2_act, in_frames * [r2_act.irrep(1)]
            )
        else:
            self.feat_type_in = e2cnn.nn.FieldType(
                r2_act, in_frames * [r2_act.regular_repr]
            )

        if self.last_layer:
            self.feat_type_hid = e2cnn.nn.FieldType(
                r2_act, out_frames * [r2_act.irrep(1)]
            )
        else:
            self.feat_type_hid = e2cnn.nn.FieldType(
                r2_act, out_frames * [r2_act.regular_repr]
            )

        if not last_layer:
            self.norm = e2cnn.nn.InnerBatchNorm(self.feat_type_hid)
            self.relu = e2cnn.nn.ReLU(self.feat_type_hid)

        grid, basis_filter, rings, sigma, maximum_frequency = compute_basis_params(
            kernel_size=kernel_size
        )
        i_repr = self.feat_type_in._unique_representations.pop()
        o_repr = self.feat_type_hid._unique_representations.pop()
        basis = self.feat_type_in.gspace.build_kernel_basis(
            i_repr, o_repr, sigma, rings, maximum_frequency=5
        )
        block_expansion = block_basisexpansion(
            basis, grid, basis_filter, recompute=False
        )

        self.basis_kernels = block_expansion.sampled_basis

        stdv = np.sqrt(1 / (in_frames * kernel_size * kernel_size))
        self.relaxed_weights = nn.Parameter(
            torch.ones(
                out_frames, self.basis_kernels.shape[0], in_frames, kernel_size**2
            ).float()
        )
        self.relaxed_weights.data.uniform_(-stdv, stdv)

        self.bias = nn.Parameter(torch.zeros(out_frames * self.basis_kernels.shape[1]))
        self.bias.data.uniform_(-stdv, stdv)

    def get_weight_constraint(self) -> Tensor:
        return torch.mean(
            torch.abs(self.relaxed_weights[..., :-1] - self.relaxed_weights[..., 1:])
        )

    def forward(self, x: Tensor) -> Tensor:
        conv_filters = torch.einsum(
            "bpqk,obik->opiqk", self.basis_kernels, self.relaxed_weights
        )
        conv_filters = conv_filters.reshape(
            conv_filters.shape[0] * conv_filters.shape[1],
            conv_filters.shape[2] * conv_filters.shape[3],
            self.kernel_size,
            self.kernel_size,
        )

        if not self.last_layer:
            out = F.conv2d(x, conv_filters, self.bias, padding=1)
            return self.relu(e2cnn.nn.GeometricTensor(out, self.feat_type_hid)).tensor
        else:
            return F.conv2d(x, conv_filters, self.bias, padding=1)
