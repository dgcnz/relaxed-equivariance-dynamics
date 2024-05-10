from torch import nn, Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
    ):
        super(ConvNet, self).__init__()
        self.layers = [ConvBlock(in_channels, hidden_dim, kernel_size)]
        self.layers += [
            ConvBlock(hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers - 2)
        ]
        self.layers += [
            nn.Conv2d(
                hidden_dim, out_channels, kernel_size, padding=(kernel_size - 1) // 2
            )
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
