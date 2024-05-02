import math
from pathlib import Path

import pytest
import torch
import torchvision.transforms.functional as TTF
from torch import Tensor

from src.utils.image_utils import rot_img

IMAGE_FOLDER = Path("tests/resources")


@pytest.mark.skip
@pytest.mark.parametrize("theta", [45])
def test_rot_img_non_straight(sample_image_tensor: Tensor, theta: float) -> None:
    """Tests `rot_img` to verify if the function behaves the same as TTF.rotate on non-straight
    angles angles."""
    # sample_image_tensor = TTF.rgb_to_grayscale(sample_image_tensor).unsqueeze(0)
    sample_image_tensor = sample_image_tensor.unsqueeze(0)
    theta_radians = math.radians(theta)
    rotated_img = rot_img(sample_image_tensor, theta_radians)
    rotated_img_ttf = TTF.rotate(sample_image_tensor, theta)
    total_pixels = sample_image_tensor.numel()
    correct_pixels = torch.isclose(rotated_img, rotated_img_ttf, atol=1e-1).int().sum()
    wrong_pixels = total_pixels - correct_pixels
    percentage_wrong = wrong_pixels / total_pixels
    assert percentage_wrong < 0.02


@pytest.mark.skip
@pytest.mark.parametrize("theta", [0, 90, -90, 450])
def test_rot_img_straight(sample_image_tensor: Tensor, theta: float) -> None:
    """Tests `rot_img` to verify if the function behaves the same as TTF.rotate on straight
    angles."""
    # unsqueeze batch dimension
    sample_image_tensor = TTF.rgb_to_grayscale(sample_image_tensor).unsqueeze(0)
    theta_radians = math.radians(theta)
    rotated_img = rot_img(sample_image_tensor, theta_radians)
    rotated_img_ttf = TTF.rotate(sample_image_tensor, theta)
    assert torch.allclose(rotated_img, rotated_img_ttf, atol=1e-4)


def test_benchmark_rot_img(benchmark, sample_image_tensor: Tensor):
    """Benchmark `rot_img`."""
    sample_image_tensor = sample_image_tensor.unsqueeze(0)
    theta_radians = math.radians(90)
    benchmark(rot_img, sample_image_tensor, theta_radians)


def test_benchmark_rot_img_ttf(benchmark, sample_image_tensor: Tensor):
    """Benchmark `TTF.rotate`."""
    sample_image_tensor = sample_image_tensor.unsqueeze(0)
    benchmark(TTF.rotate, sample_image_tensor, 90)
