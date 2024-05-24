import torch
import numpy as np 
from src.utils.image_utils import rot_field


def get_equivariance_error(model, datamodule, device):
    equiv_errors = []
    test_loader = datamodule.test_dataloader()
    

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            #print(xx.shape)
            orig_pred = model(xx).reshape(-1, 2, xx.shape[-2], xx.shape[-1])

            for angle in [np.pi/2, np.pi, np.pi/2*3]:
                rho_inp = rot_field(xx.reshape(-1, 2, xx.shape[-2], xx.shape[-1]), angle).to(device)
                rho_inp = rho_inp.reshape(datamodule.batch_size_per_device, -1, xx.shape[-2], xx.shape[-1])
                #error line below 
                rho_inp_outs = model(rho_inp).reshape(-1, 2, xx.shape[-2], xx.shape[-1])
                equiv_errors.append(torch.mean(torch.abs(rho_inp_outs - rot_field(orig_pred, angle))).data.cpu())
    
    equiv_error = np.mean(equiv_errors)
    return equiv_error