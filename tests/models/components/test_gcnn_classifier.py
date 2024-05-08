from torch import Tensor
import torchvision.transforms.functional as TTF
from src.models.components.gcnn_classifier import GCNNClassifier
from PIL import Image


def test_gcnn_classifier_forward(sample_image: Image):
    image_batch = TTF.to_tensor(sample_image).unsqueeze(0)
    # get image dimensions
    B, Cin, H, W = image_batch.shape
    # create a GCNNClassifier model
    model = GCNNClassifier(
        in_channels=Cin,
        kernel_size=3,
        hidden_dim=1,
        group_order=4,
        num_gconvs=2,
        num_classes=10,
    )
    # perform a forward pass
    output = model(image_batch)
    # check that the output has the correct shape
    assert output.shape == (B, 10)
