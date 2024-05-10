from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.smokeplume_dataset import SmokePlumeDataset
import logging
import itertools

logging.basicConfig(level=logging.INFO)


class SmokePlumeDataModule(LightningDataModule):
    def __init__(
        self,
        equivariance_level: int,
        input_length: int = 1,
        mid: int = 3,
        output_length: int = 6,
        root_dir: str = "data/",
        task_list: list[int] = [0, 1, 2, 3],
        train_val_test_split: tuple[int, int, int] = (30, 10, 0),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        assert train_val_test_split[0] > 0 and train_val_test_split[1] > 0

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.logger = logging.getLogger(__name__)

        # data transformations
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        SmokePlumeDataset.download_and_extract(root_dir=self.hparams.root_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            num = self.hparams.train_val_test_split
            accnum = list(itertools.accumulate(num))
            train_times = list(range(accnum[0]))
            val_times = list(range(accnum[0], accnum[1]))
            test_split = list(range(accnum[1], accnum[2]))
            if num[2] == 0:
                test_split = val_times
                self.logger.warn(
                    "No test data provided. Using validation data for testing."
                )

            params = {
                "root": self.hparams.root_dir,
                "input_length": self.hparams.input_length,
                "mid": self.hparams.mid,
                "output_length": self.hparams.output_length,
                "task_list": self.hparams.task_list,
                "equivariance_level": self.hparams.equivariance_level,
            }

            self.data_train = SmokePlumeDataset(**params, sample_list=train_times)
            self.data_val = SmokePlumeDataset(**params, sample_list=val_times)
            self.data_test = SmokePlumeDataset(**params, sample_list=test_split)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = SmokePlumeDataModule(
        equivariance_level=0,
    )
    dm.prepare_data()
