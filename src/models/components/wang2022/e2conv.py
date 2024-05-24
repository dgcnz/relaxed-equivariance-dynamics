# Source:
# https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets/blob/e8e9355ebd241d5cbe010a8a76ce6a7f52d1691f/models/model_rotation.py
import e2cnn
import torch


class E2Conv(torch.nn.Module):
    def __init__(self, in_frames: int, out_frames: int, kernel_size: int, N: int):
        super(E2Conv, self).__init__()

        r2_act = e2cnn.gspaces.Rot2dOnR2(N=N)
        feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames * [r2_act.regular_repr])
        feat_type_hid = e2cnn.nn.FieldType(r2_act, out_frames * [r2_act.regular_repr])

        self.layer = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(
                feat_type_in,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            e2cnn.nn.InnerBatchNorm(feat_type_hid),
            e2cnn.nn.ReLU(feat_type_hid),
        )

    def forward(self, xx):
        return self.layer(xx)


class E2CNN(torch.nn.Module):
    def __init__(
        self,
        in_frames: int,
        out_frames: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        N: int,
    ):
        super(E2CNN, self).__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N=N)

        self.feat_type_in = e2cnn.nn.FieldType(r2_act, in_frames * [r2_act.irrep(1)])
        self.feat_type_hid = e2cnn.nn.FieldType(
            r2_act, hidden_dim * [r2_act.regular_repr]
        )
        self.feat_type_out = e2cnn.nn.FieldType(r2_act, out_frames * [r2_act.irrep(1)])

        input_layer = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(
                self.feat_type_in,
                self.feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            e2cnn.nn.InnerBatchNorm(self.feat_type_hid),
            e2cnn.nn.ReLU(self.feat_type_hid),
        )

        layers = [input_layer]
        layers += [
            E2Conv(hidden_dim, hidden_dim, kernel_size, N)
            for i in range(num_layers - 2)
        ]
        layers += [
            e2cnn.nn.R2Conv(
                self.feat_type_hid,
                self.feat_type_out,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, xx):
        xx = e2cnn.nn.GeometricTensor(xx, self.feat_type_in)
        out = self.model(xx)
        return out.tensor
