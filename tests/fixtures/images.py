import pytest
import torchvision.transforms.functional as TTF
from PIL import Image
from torch import Tensor


@pytest.fixture()
def sample_image() -> Image:
    return Image.open("tests/resources/sample_image.jpg").convert("RGB")


@pytest.fixture()
def sample_image_tensor(sample_image) -> Tensor:
    return TTF.to_tensor(sample_image)
