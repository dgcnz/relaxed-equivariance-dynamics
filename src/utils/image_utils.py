# TODO: add LICENSE / CITATION
import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor, Tensor


def rot_img(x: Tensor, theta: float) -> Tensor:
    """Rotate batch of images by `theta` radians.

    :param x: batch of images with shape [N, C, H, W]
    :param theta: angle :returns rotated images
    """
    # Rotation Matrix (2 x 3)
    rot_mat = FloatTensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
        ]
    )

    # The affine transformation matrices should have the shape of N x 2 x 3
    rot_mat = rot_mat.repeat(x.shape[0], 1, 1)

    # Obtain transformed grid
    # grid is the coordinates of pixels for rotated image
    # F.affine_grid assumes the origin is in the middle
    # and it rotates the positions of the coordinates
    # r(f(x)) = f(r^-1 x)
    grid = F.affine_grid(rot_mat.to(x.device), x.size(), align_corners=False).float()
    x = F.grid_sample(x, grid)
    return x.float()
