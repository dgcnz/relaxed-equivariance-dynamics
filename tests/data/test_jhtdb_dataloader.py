import pytest
import torch

from src.data.jhtdb_datamodule import JHTDBDataModule


@pytest.mark.parametrize(
    "config_name,lr_dim,hr_dim,ws,num", [("small_50", 4, 16, 3, 50)]
)
def test_mnist_datamodule(
    config_name: str, lr_dim: int, hr_dim: int, ws: int, num: int
) -> None:
    BATCH_SIZE = 4
    dm = JHTDBDataModule(batch_size=BATCH_SIZE, dataset_config_name=config_name)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == num

    batch = next(iter(dm.test_dataloader()))
    lrs = batch["lrs"]
    hr = batch["hr"]
    assert lrs.shape == (BATCH_SIZE, ws, 3, lr_dim, lr_dim, lr_dim)
    assert hr.shape == (BATCH_SIZE, 3, hr_dim, hr_dim, hr_dim)
    assert lrs.dtype == torch.float32
    assert hr.dtype == torch.float32
