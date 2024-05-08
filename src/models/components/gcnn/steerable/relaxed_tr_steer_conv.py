import torch 
from src.models.components.gcnn.steerable.relaxed_r_steer_conv import (
    Relaxed_R_SteerConv,
)
import e2cnn
import torch.nn as nn


class Relaxed_TR_SteerConv(nn.Module):
    def __init__(self, in_frames, out_frames, kernel_size, N, num_banks, h_size, w_size, first_layer = False, last_layer = False):
        super(Relaxed_TR_SteerConv, self).__init__()
        self.convs = nn.Sequential(*[Relaxed_R_SteerConv(in_frames = in_frames, out_frames = out_frames, 
                                                       kernel_size = kernel_size, N = N, first_layer = first_layer, 
                                                       last_layer = last_layer) for i in range(num_banks)])
        
        self.combination_weights = nn.Parameter(torch.ones(h_size, w_size, num_banks).float()/num_banks)
        
        #self.activation = nn.ReLU()
        self.kernel_size = kernel_size
        self.pad_size = (kernel_size-1)//2
        self.h_size = h_size
        self.w_size = w_size
        self.last_layer = last_layer
        self.num_banks = num_banks
            

    def get_weight_constraint(self):
        return sum([layer.get_weight_constraint() for layer in self.convs])
        
    def forward(self, x):
        outs = torch.stack([self.convs[i](x) for i in range(self.num_banks)], dim  = 0)
        
        # Compute Convolution
        out = torch.einsum("ijr, rboij -> boij", self.combination_weights, outs)
        
        
        if self.last_layer:
            return out
        else:
            return self.convs[0].relu(e2cnn.nn.GeometricTensor(out, self.convs[0].feat_type_hid)).tensor