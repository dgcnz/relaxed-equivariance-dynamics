import pytest
from PIL import Image
import torchvision.transforms.functional as TTF
from torch import Tensor


@pytest.fixture()
def sample_image() -> Image:
    return Image.open("tests/resources/sample_image.jpg").convert("RGB")

@pytest.fixture()
def sample_image_tensor(sample_image) -> Tensor:
    return TTF.to_tensor(sample_image)