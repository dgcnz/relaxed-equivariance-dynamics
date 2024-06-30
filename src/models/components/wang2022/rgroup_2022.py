# Source:
# https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets/blob/e8e9355ebd241d5cbe010a8a76ce6a7f52d1691f/models/model_rotation.py
import torch
import numpy as np
import torch.nn.functional as F
import copy
import os
import numpy as np
import matplotlib.pyplot as plt


class RelaxedGroupEquivariantCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,
        group_order: int,
        num_gconvs: int,
        num_filter_banks: int,
        vel_inp: bool = True,
        alpha: float = 0.0,
    ):
        super().__init__()

        # First transform \rho_1 to regular representations.
        self.alpha = alpha
        theta = torch.tensor(2 * np.pi / group_order).float()
        self.register_buffer(
            "lift_coefs",
            torch.tensor(
                [
                    [torch.cos(theta * i), torch.sin(theta * i)]
                    for i in range(group_order)
                ]
            ).float(),
        )

        print(f"vel_inp: {vel_inp}")

        if vel_inp:
            self.gconvs = [
                Relaxed_GroupConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    group_order=group_order,
                    num_filter_banks=num_filter_banks,
                    activation=True,
                    save_image=True
                )
            ]
        else:
            self.gconvs = [
                Relaxed_LiftingConvolution(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    group_order=group_order,
                    num_filter_banks=num_filter_banks,
                    activation=True,
                    save_image=True
                )
            ]

        for i in range(num_gconvs - 2):
            self.gconvs.append(
                Relaxed_GroupConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    group_order=group_order,
                    num_filter_banks=num_filter_banks,
                    activation=True,
                )
            )

        self.gconvs.append(
            Relaxed_GroupConv(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                group_order=group_order,
                num_filter_banks=num_filter_banks,
                activation=False,
            )
        )

        self.gconvs = torch.nn.Sequential(*self.gconvs)

        self.vel_inp = vel_inp
        self.group_order = group_order

    def get_weight_constraint(self) -> float:
        return self.alpha * sum(
            [gconv.get_weight_constraint() for gconv in self.gconvs]
        )

    def forward(self, x, target_length=1):
        if self.vel_inp and len(x.shape) == 4:
            # x: [8,2, 64, 64]
            x = x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2], x.shape[3])
            # x: [8, 1, 2, 64, 64]
        preds = []
        for i in range(target_length):
            if self.vel_inp:
                z = torch.einsum("bivhw, nv->binhw", x, self.lift_coefs)
                # x: [8, 1, 4, 64, 64]
            else:
                z = x
            out = self.gconvs(z)
            if self.vel_inp:
                # x: [8, 1, 4, 64, 64]
                out = torch.einsum("binhw, nv->bivhw", out, self.lift_coefs)
                # x: [8, 1, 2, 64, 64]
            else:
                out = out.mean(2)
            x = torch.cat([x[:, out.shape[1] :], out], 1)
            preds.append(out)

        outs = torch.cat(preds, dim=1)
        outs = outs.reshape(outs.shape[0], -1, outs.shape[-2], outs.shape[-1])
        return outs


class Relaxed_GroupConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_order,
        num_filter_banks,
        activation=True,
        save_image = True
    ):

        super(Relaxed_GroupConv, self).__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation
        self.save_image = save_image
        self.batch_number = 0

        ## Initialize weights
        self.combination_weights = torch.nn.Parameter(
            torch.ones(group_order, num_filter_banks).float()
            / num_filter_banks
            / group_order
        )
        self.weight = torch.nn.Parameter(
            torch.randn(
                self.num_filter_banks,  ##additional dimension
                self.out_channels,
                self.in_channels,
                self.group_order,
                self.kernel_size,
                self.kernel_size,
            )
        )

        stdv = np.sqrt(1 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.weight.data.uniform_(-stdv, stdv)

        # If combination_weights are equal values, then the model is still equivariant
        # self.combination_weights.data.uniform_(-stdv, stdv)

    def get_weight_constraint(self) -> float:
        return get_weight_constraint(self.combination_weights.transpose(0, 1))

    def generate_filter_bank(self):
        """Obtain a stack of rotated and cyclic shifted filters"""
        filter_bank = []
        weights = self.weight.reshape(
            self.num_filter_banks * self.out_channels * self.in_channels,
            self.group_order,
            self.kernel_size,
            self.kernel_size,
        )

        for i in range(self.group_order):
            # planar rotation
            rotated_filter = rot_img(weights, -np.pi * 2 / self.group_order * i)

            # cyclic shift
            shifted_indices = torch.roll(torch.arange(0, self.group_order, 1), shifts=i)
            shifted_rotated_filter = rotated_filter[:, shifted_indices]

            filter_bank.append(
                shifted_rotated_filter.reshape(
                    self.num_filter_banks,
                    self.out_channels,
                    self.in_channels,
                    self.group_order,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        # stack
        filter_bank = torch.stack(filter_bank).permute(1, 2, 0, 3, 4, 5, 6)
        return filter_bank

    def forward(self, x):

        # input shape: [bz, #in, group_order, h, w]
        x_in = x
        self.batch_number += 1

        filter_bank = self.generate_filter_bank()

        relaxed_conv_weights = torch.einsum(
            "na, aon... -> on...", self.combination_weights, filter_bank
        )

        x = torch.nn.functional.conv2d(
            input=x.reshape(
                x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]
            ),
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels * self.group_order,
                self.kernel_size,
                self.kernel_size,
            ),
            padding=(self.kernel_size - 1) // 2,
        )

        # Reshape signal back [bz, #out * g_order, h, w] -> [bz, out, g_order, h, w]
        x = x.view(
            x.shape[0], self.out_channels, self.group_order, x.shape[-2], x.shape[-1]
        )
        # ========================
        
        if self.save_image:
            if self.batch_number > (140*8):
                #also do a forward where the relaxed weights are all one
                relaxed_conv_weights = torch.ones_like(relaxed_conv_weights)
                x_no_relax = torch.nn.functional.conv2d(
                    input=x.reshape(
                        x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]
                    ),
                    weight=relaxed_conv_weights.reshape(
                        self.out_channels * self.group_order,
                        self.in_channels * self.group_order,
                        self.kernel_size,
                        self.kernel_size,
                    ),
                    padding=(self.kernel_size - 1) // 2,
                )

                # Reshape signal back [bz, #out * g_order, h, w] -> [bz, out, g_order, h, w]
                x_no_relax = x_no_relax.view(
                    x_no_relax.shape[0], self.out_channels, self.group_order, x_no_relax.shape[-2], x_no_relax.shape[-1]
                )
                
                self.save_channel_magnitude_image(x_no_relax, output_dir=os.path.join('images', 'no_relax_output'))
                self.save_channel_magnitude_image(x_in, output_dir=os.path.join('images', 'input'))
                self.save_channel_magnitude_image(x, output_dir=os.path.join('images', 'output'))


        return x
    
    def save_channel_magnitude_image(self, tensor, output_dir='images'):
        """
        Calculate the magnitude of the output channel vectors and save the resulting image.

        Parameters:
        tensor (numpy.ndarray): Input tensor of shape [bz, #out, group order, h, w]
        output_dir (str): Directory to save the resulting image
        """
        # Ensure the output directory exists
        output_dir = os.path.join(output_dir, f'batch_{self.batch_number}')
        os.makedirs(output_dir, exist_ok=True)
        #print('saved some images')

        # Select the first batch dimension
        for order in range(tensor.shape[2]):
            order_tensor = tensor[:, :, order, :, :]
            tensor_first_batch = order_tensor[0]  # Shape: [#out, group order, h, w]

            tensor_first_batch = tensor_first_batch.detach().cpu().numpy()

            # Compute the magnitude of the vectors across the output channels
            magnitude = np.linalg.norm(tensor_first_batch, axis=0)

            # Normalize the magnitude to the range [0, 255]
            magnitude_min = np.min(magnitude)
            magnitude_max = np.max(magnitude)
            magnitude_normalized = 255 * (magnitude - magnitude_min) / (magnitude_max - magnitude_min)
            magnitude_normalized = magnitude_normalized.astype(np.uint8)

            # Save the resulting image
            output_path = os.path.join(output_dir, f'magnitude_image{order}.png')
            plt.imsave(output_path, magnitude_normalized, cmap='Reds')


class Relaxed_LiftingConvolution(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_order,
        num_filter_banks,
        activation=True,
        save_image = True
    ):
        super(Relaxed_LiftingConvolution, self).__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation
        self.save_image = save_image
        self.batch_number = 0

        self.combination_weights = torch.nn.Parameter(
            torch.ones(num_filter_banks, group_order).float() / num_filter_banks
        )

        # Initialize an unconstrained kernel.
        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.num_filter_banks,  # Additional dimension
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        stdv = np.sqrt(1 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.weight.data.uniform_(-stdv, stdv)

        # If combination_weights are equal values, then the model is still equivariant
        # self.combination_weights.data.uniform_(-stdv, stdv)

    def get_weight_constraint(self) -> float:
        return get_weight_constraint(self.combination_weights)

    def generate_filter_bank(self):
        """Obtain a stack of rotated filters"""
        weights = self.weight.reshape(
            self.num_filter_banks * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        filter_bank = torch.stack(
            [
                rot_img(weights, -np.pi * 2 / self.group_order * i)
                for i in range(self.group_order)
            ]
        )
        filter_bank = filter_bank.transpose(0, 1).reshape(
            self.num_filter_banks,  # Additional dimension
            self.out_channels,
            self.group_order,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        return filter_bank

    def forward(self, x):
        # input shape: [bz, #in, h, w]
        # output shape: [bz, #out, group order, h, w]

        # generate filter bank given input group order
        self.batch_number += 1
        x_in = copy.deepcopy(x)
        filter_bank = self.generate_filter_bank()

        # for each rotation, we have a linear combination of multiple filters with different coefficients.
        relaxed_conv_weights = torch.einsum(
            "na, noa... -> oa...", self.combination_weights, filter_bank
        )

        # concatenate the first two dims before convolution.
        # ==============================
        x = F.conv2d(
            input=x,
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ),
            padding=(self.kernel_size - 1) // 2,
        )
        # ==============================

        # reshape output signal to shape [bz, #out, group order, h, w].
        # ==============================
        x = x.view(
            x.shape[0], self.out_channels, self.group_order, x.shape[-1], x.shape[-2]
        )
        # ==============================

        if self.save_image:
            self.save_channel_magnitude_image(x_in, output_dir=os.path.join('images', 'input'))
            self.save_channel_magnitude_image(x, output_dir=os.path.join('images', 'output'))


        if self.activation:
            return F.relu(x)
        return x
    
    def save_channel_magnitude_image(self, tensor, output_dir='images'):
        """
        Calculate the magnitude of the output channel vectors and save the resulting image.

        Parameters:
        tensor (numpy.ndarray): Input tensor of shape [bz, #out, group order, h, w]
        output_dir (str): Directory to save the resulting image
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print('saved some images')

        # Select the first batch dimension
        for order in range(tensor.shape[2]):
            order_tensor = tensor[:, :, order, :, :]
            tensor_first_batch = order_tensor[0]  # Shape: [#out, group order, h, w]

            # Compute the magnitude of the vectors across the output channels
            magnitude = np.linalg.norm(tensor_first_batch, axis=0)

            # Normalize the magnitude to the range [0, 255]
            magnitude_min = np.min(magnitude)
            magnitude_max = np.max(magnitude)
            magnitude_normalized = 255 * (magnitude - magnitude_min) / (magnitude_max - magnitude_min)
            magnitude_normalized = magnitude_normalized.astype(np.uint8)

            # Save the resulting image
            output_path = os.path.join(output_dir, f'magnitude_image{order}-{self.batch_number}.png')
            plt.imsave(output_path, magnitude_normalized, cmap='Reds')


def get_weight_constraint(w: torch.Tensor):
    w = w.unsqueeze(2)
    return torch.triu(w - w.transpose(1, 2)).abs().sum()


def rot_img(x, theta):
    """Rotate 2D images
    Args:
        x : input images with shape [N, C, H, W]
        theta: angle
    Returns:
        rotated images
    """
    # Rotation Matrix (2 x 3)
    rot_mat = torch.FloatTensor(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]]
    ).to(x.device)

    # The affine transformation matrices should have the shape of N x 2 x 3
    rot_mat = rot_mat.repeat(x.shape[0], 1, 1)

    # Obtain transformed grid
    # grid is the coordinates of pixels for rotated image
    # F.affine_grid assumes the origin is in the middle
    # and it rotates the positions of the coordinates
    # r(f(x)) = f(r^-1 x)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False).float().to(x.device)
    x = F.grid_sample(x, grid)
    return x

