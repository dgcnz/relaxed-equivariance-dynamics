import numpy as np
import torchvision.transforms.functional as TTF
from torch import Tensor
import torch


def generate_rot_filter_bank(kernel: Tensor, group_order: int) -> Tensor:
    """Generate a stack of rotated filters
    :param kernel: the kernel tensor of shape [#out, #in, k, k]
    :param group_order: the order of the rotation group (4 = C4)
    :return: a tensor of shape [#out, group_order, #in, k, k]
    """
    thetas = np.arange(group_order) * (-360 / group_order)
    return torch.stack([TTF.rotate(kernel, theta) for theta in thetas], dim=1)
