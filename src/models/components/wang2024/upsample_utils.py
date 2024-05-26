from torch import nn, Tensor


class GroupUpsampler3D(nn.Module):
    def __init__(self, upsampler: nn.Upsample) -> None:
        super().__init__()
        self.upsampler = upsampler

    def forward(self, input: Tensor) -> Tensor:
        # TODO make some unit tests for this, since it is sketch
        # TODO look at if we want to combine batch and group or channel and group (currently batch and group)
        batch, channel, group, x, y, z = input.shape

        input = input.permute(0, 2, 1, 3, 4, 5)  # Put batch and group next to eacother

        input = input.reshape(
            -1, channel, x, y, z
        )  # Treats every group as a seperate element of the batch

        output: Tensor = self.upsampler(input)

        _, _, newx, newy, newz = (
            output.shape
        )  # Read this out explicitly since it can depend on the upscaler

        output = output.reshape(batch, group, channel, newx, newy, newz)

        return output.permute(0, 2, 1, 3, 4, 5)  # Put back to original order
