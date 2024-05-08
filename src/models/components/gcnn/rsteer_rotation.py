import torch
from src.models.components.gcnn.steerable.relaxed_rotation_steer import (
    Relaxed_Rot_SteerConv,
)

class Relaxed_Rot_SteerConvNet(torch.nn.Module):
    def __init__(self, in_frames, out_frames, hidden_dim, kernel_size, num_layers, N, alpha = 1):
        super(Relaxed_Rot_SteerConvNet, self).__init__()
        self.alpha = alpha

        layers = [Relaxed_Rot_SteerConv(in_frames = in_frames, out_frames = hidden_dim, 
                                 kernel_size = kernel_size, N = N, 
                                 first_layer = True, last_layer = False)]
        
        layers += [Relaxed_Rot_SteerConv(in_frames = hidden_dim, out_frames = hidden_dim, 
                                  kernel_size = kernel_size, N = N, 
                                  first_layer = False, last_layer = False) 
                   for i in range(num_layers-2)]
        
        layers += [Relaxed_Rot_SteerConv(in_frames = hidden_dim, out_frames = out_frames, 
                                  kernel_size = kernel_size, N = N, 
                                  first_layer = False, last_layer = True) ]
        self.model = torch.nn.Sequential(*layers)
        
    def rot_vector(self, inp, theta):
        #inp shape: c x 2 x 64 x 64
        theta = torch.tensor(theta).float()
        rot_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).float()
        out = torch.einsum("ab, bc... -> ac...",(rot_matrix, inp.transpose(0,1))).transpose(0,1)
        return out
    
    def get_rot_mat(self, theta):
        theta = torch.tensor(theta).float()
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]]).float()

    def rot_img(self, x, theta):
        rot_mat = self.get_rot_mat(theta)[None, ...].float().repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).float()
        x = F.grid_sample(x, grid)
        return x.float()
   
    def get_weight_constraint(self):
        return self.alpha * sum([layer.get_weight_constraint() for layer in self.model])
    
        
    def forward(self, xx):
        return self.model(xx)