import torch
from src.utils.lie_utils import *

def jvp(f, x, u):
    """Jacobian vector product Df(x)u vs typical autograd VJP vTDF(x).
    Uses two backwards passes: computes (vTDF(x))u and then derivative wrt to v to get DF(x)u"""
    with torch.enable_grad():
        y = f(x)
        v = torch.ones_like(
            y, requires_grad=True
        )  # Dummy variable (could take any value)
        vJ = torch.autograd.grad(y, [x], [v], create_graph=True)
        Ju = torch.autograd.grad(vJ, [v], [u], create_graph=True)
        return Ju[0]
    
def equiv_err_derivative(model, inp_imgs, type):
    if type == 'rotation':
        def rotated_model(t):
            rotated_img = rotate(inp_imgs, t)
            z = model(rotated_img)
            if img_like(z.shape):
                z = rotate(z, -t)
            return z

        t = torch.zeros(1, requires_grad=True, device=inp_imgs.device)
        lie_deriv = jvp(rotated_model, t, torch.ones_like(t))

    equiv_err = torch.norm(lie_deriv.view(lie_deriv.size(0), -1), dim=(1)).mean()/(len(lie_deriv.size())-1)

    return equiv_err