import torch
from src.utils.lie_utils import *
    
def get_lie_equiv_err(model, datamodule, device, type='rotation'):
    equiv_err = 0

    if type == 'rotation':
        def rotated_model(t):
            rotated_img = rotate(inp_imgs, t)
            rotated_img = rotated_img.to(device)
            z = model(rotated_img)
            if img_like(z.shape):
                z = rotate(z, -t)
            return z

        for (inp_imgs, _) in datamodule.test_dataloader():
            inp_imgs = inp_imgs.to(device)
            t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
            lie_deriv = jvp(rotated_model, t, torch.ones_like(t))
            equiv_err += torch.norm(lie_deriv.view(lie_deriv.size(0), -1), dim=(1)).mean()/(len(lie_deriv.size())-1)

    return equiv_err.cpu().detach().numpy()