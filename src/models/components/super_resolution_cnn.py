from torch import Tensor
import torch.nn as nn

class SuperResCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_dim: int,        
        
        
        
    ):
        
        super().__init__()
        self.upsampler = nn.Upsample(scale_factor=2,mode='trilinear')

        self.first_conv = nn.Conv3d(in_channels,hidden_dim,kernel_size,stride=1,padding=kernel_size//2,dilation=1)

        self.upconv_block1= nn.Sequential(
            nn.ConvTranspose3d(hidden_dim,hidden_dim,kernel_size,stride=2,padding=kernel_size//2,output_padding=1,dilation=1),
            nn.Conv3d(hidden_dim,hidden_dim,kernel_size,stride=1,padding=kernel_size//2,dilation=1)
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(hidden_dim,hidden_dim,kernel_size,stride=1,padding=kernel_size//2,dilation=1),
            nn.Conv3d(hidden_dim,hidden_dim,kernel_size,stride=1,padding=kernel_size//2,dilation=1)

        )

        self.upconv_block2= nn.Sequential(
            nn.ConvTranspose3d(hidden_dim,hidden_dim,kernel_size,stride=2,padding=kernel_size//2,output_padding=1,dilation=1),
            nn.Conv3d(hidden_dim,hidden_dim,kernel_size,stride=1,padding=kernel_size//2,dilation=1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(hidden_dim,hidden_dim,kernel_size,stride=1,padding=kernel_size//2,dilation=1),
            nn.Conv3d(hidden_dim,hidden_dim,kernel_size,stride=1,padding=kernel_size//2,dilation=1)

        )

        self.final_conv = nn.Conv3d(hidden_dim,out_channels,kernel_size,stride=1,padding=kernel_size//2,dilation=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.first_conv(x)
        x2 = self.upconv_block1(x1)
        x2 += self.upsampler(x1)
        x3 = self.conv_block1(x2)
        x3 += x2
        x4 = self.upconv_block2(x3)
        x4 += self.upsampler(x3)
        x5 = self.conv_block2(x4)
        x5 += x4
        x6 = self.final_conv(x5)

        return x6
