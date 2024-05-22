from gconv.gnn import (
    GSeparableConvSE3,
    GLiftingConvSE3,
    RGLiftingConvSE3,
    RGSeparableConvSE3,
)
from gconv.geometry.groups import so3
from torch import Tensor
import torch.nn as nn

class GroupUpsampler3D(nn.Module):
    def __init__(self, upsampler) -> None:
        super().__init__()
        self.upsampler = upsampler
    def forward(self,input):
        batch,channel,group,x,y,z = input.shape

        input= input.permute(0,2,1,3,4,5) # Put batch and group next to eacother

        input = input.reshape(-1,channel,x,y,z) #Treats every group as a seperate element of the batch

        output = self.upsampler(input)

        _,_,newx,newy,newz = output.shape #Read this out explicitly since it can depend on the upscaler

        output = output.reshape(batch,group,channel,newx,newy,newz)

        return output.permute(0,2,1,3,4,5) # Put back to original order

class GCNNOhT3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,        
        classifier: bool = False,
        sigmoid: bool = False,
        
    ):
        
        super().__init__()

        self.classifier = classifier
        self.sigmoid = sigmoid
        self.grid_Oh = so3.quat_to_matrix(so3.octahedral())
        #self.dims = [in_channels] + [hidden_dim] * (num_gconvs - 2) + [out_channels]
        self.upsampler = GroupUpsampler3D(nn.Upsample(scale_factor=2,mode='trilinear'))

        self.lift = GLiftingConvSE3(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            grid_H=self.grid_Oh,
            permute_output_grid=False,
            mask=False,
        )
        
        self.upconv_block1 = nn.ModuleList([
            GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,
                    conv_mode='3d_transposed',
                    output_padding=1,
                    stride=2,
                ),
            GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,                    
                ),

        ])

        self.upconv_block2 = nn.ModuleList([
            GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,
                    conv_mode='3d_transposed',
                    output_padding=1,
                    stride=2,
                ),
            GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,                    
                ),

        ])

        self.conv_block1 = nn.ModuleList([
            GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,
                ),
            GSeparableConvSE3(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                permute_output_grid=False,
                grid_H=self.grid_Oh,
                mask=False,
            )

        ])
        self.conv_block2 = nn.ModuleList([
            GSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,
                ),
            GSeparableConvSE3(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                permute_output_grid=False,
                grid_H=self.grid_Oh,
                mask=False,
            )

        ])

        self.final_conv = GSeparableConvSE3(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                permute_output_grid=False,
                grid_H=self.grid_Oh,
                mask=False,
            )
        
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, H, W]
        """
        z1, H1 = self.lift(x, self.grid_Oh)     
           
        
        #The H's never scale so we don't need to upsample them
        #Do we need to add them to the skip connection though?
        z2 = z1.clone()
        H2 = H1.clone()
        
        for gconv in self.upconv_block1:
            
            z2,H2 = gconv(z2,H2)
            
        
        
        
        z2 += self.upsampler(z1)

        z3 = z2.clone()
        H3 = H2.clone()
        for gconv in self.conv_block1:
            
            z3, H3 = gconv(z3,H3)
               
        z3 += z2

        z4 = z3.clone()
        H4 = H3.clone()
        for gconv in self.upconv_block2:
            z4, H4 = gconv(z4,H4)
        z4 += self.upsampler(z3)

        z5 = z4.clone()
        H5 = H5.clone()
        for gconv in self.conv_block2:
            z5, H5 = gconv(z5,H5)        
        z5 += z4

        z6, H6 = self.final_conv(z5,H5)        

        #These parts might need some pooling or whatever
        #Like in the gconv tutorial
        if self.classifier:
            z6 = z6.mean((2, 3, 4, 5))
        if self.sigmoid:
            z6 = z6.sigmoid()
        return z6


class RGCNNOhT3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filter_banks: int,
        kernel_size: int,
        hidden_dim: int,
        num_gconvs: int,
        classifier: bool = False,
        sigmoid: bool = False,
    ):
        assert num_gconvs >= 2
        super().__init__()

        self.classifier = classifier
        self.sigmoid = sigmoid
        self.grid_Oh = so3.quat_to_matrix(so3.octahedral())
        self.dims = [in_channels] + [hidden_dim] * (num_gconvs - 2) + [out_channels]
        self.lift = RGLiftingConvSE3(
            in_channels=in_channels,
            out_channels=hidden_dim,
            num_filter_banks=num_filter_banks,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            grid_H=self.grid_Oh,
            permute_output_grid=False,
            mask=False,
        )
        self.gconvs = nn.ModuleList(
            [
                RGSeparableConvSE3(
                    in_channels=hidden_dim,
                    out_channels=self.dims[i],
                    num_filter_banks=num_filter_banks,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    permute_output_grid=False,
                    grid_H=self.grid_Oh,
                    mask=False,
                )
                for i in range(1, num_gconvs)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the group convolution layer
        :param x: input tensor of shape [B, #in, H, W]
        """
        z, H = self.lift(x, self.grid_Oh)
        for gconv in self.gconvs:
            z, H = gconv(z, H)
        if self.classifier:
            z = z.mean((2, 3, 4, 5))
        if self.sigmoid:
            z = z.sigmoid()
        return z
