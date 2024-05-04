import torch
import torch.nn.functional as F
import math
import numpy as np
from src.utils.image_utils import rot_img

##### 2D Relaxed Rotation Group Convolution Layer #####
class RuiRelaxedRotGroupConv2d(torch.nn.Module):
    """Relaxed group convolution Layer for 2D finite rotation group"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 group_order, # the order of 2d finite rotation group
                 num_filter_banks,
                 activation = True # whether to apply relu in the end
                ):

        super().__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation


        # Initialize weights
        # If relaxed_weights are equal values, then the model is still equivariant
        # Relaxed weights do not need to be equal across different filter bank
        self.relaxed_weights = torch.nn.Parameter(torch.ones(group_order, num_filter_banks).float())
        self.kernel = torch.nn.Parameter(torch.randn(self.num_filter_banks, # additional dimension
                                                     self.out_channels,
                                                     self.in_channels,
                                                     self.group_order,
                                                     self.kernel_size,
                                                     self.kernel_size))

        torch.nn.init.kaiming_uniform_(self.kernel.data, a=math.sqrt(5))

        
    def generate_filter_bank(self):
        """ Obtain a stack of rotated and cyclic shifted filters"""
        filter_bank = []
        weights = self.kernel.reshape(self.num_filter_banks*self.out_channels*self.in_channels,
                                      self.group_order,
                                      self.kernel_size,
                                      self.kernel_size)

        for i in range(self.group_order):
            # planar rotation
            rotated_filter = rot_img(weights, -np.pi*2/self.group_order*i)

            # cyclic shift
            shifted_indices = torch.roll(torch.arange(0, self.group_order, 1), shifts = i)
            shifted_rotated_filter = rotated_filter[:,shifted_indices]


            filter_bank.append(shifted_rotated_filter.reshape(self.num_filter_banks,
                                                              self.out_channels,
                                                              self.in_channels,
                                                              self.group_order,
                                                              self.kernel_size,
                                                              self.kernel_size))
        # stack
        filter_bank = torch.stack(filter_bank).permute(1,2,0,3,4,5,6)
        return filter_bank
    
    def forward(self, x):

        filter_bank = self.generate_filter_bank()

        relaxed_conv_weights = torch.einsum("na, aon... -> on...", self.relaxed_weights, filter_bank)

        x = torch.nn.functional.conv2d(
            input=x.reshape(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                x.shape[4]
                ),
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels * self.group_order,
                self.kernel_size,
                self.kernel_size
            ),
            padding = (self.kernel_size-1)//2
        )

        # Reshape signal back [bz, #out * g_order, h, w] -> [bz, out, g_order, h, w]
        x = x.view(x.shape[0], self.out_channels, self.group_order, x.shape[-2], x.shape[-1])
        # ========================
        if self.activation:
            return F.leaky_relu(x)
        return x
