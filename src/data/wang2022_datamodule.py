from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.wang2022_dataset import Wang2022Dataset
import logging
import itertools

logging.basicConfig(level=logging.INFO)


class Wang2022Datamodule(LightningDataModule):
    def __init__(
        self,
        symmetry: str,
        future: bool = False,
        input_length: int = 1,
        mid: int = 3,
        output_length: int = 6,
        root_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
    
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.logger = logging.getLogger(__name__)

        self.train_task, self.train_time, self.val_task, self.val_time, self.test_domain_task, self.test_domain_time, self.test_future_task, self.test_future_time = self.get_task_and_sample_list(self.hparams.symmetry)

        # data transformations
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
    
    def get_task_and_sample_list(self, symmetry: str):
        train_time = list(range(0, 160))
        valid_time = list(range(160, 200))
        test_future_time = list(range(200, 250))
        test_domain_time = list(range(0, 100))

        if symmetry == "Translation":
            train_task = [(48, 10), (56, 10), (8, 20),  (40, 5),  (56, 25), 
                        (48, 20), (48, 5),  (16, 20), (56, 5),  (32, 10), 
                        (56, 15), (16, 5),  (40, 15), (40, 25), (48, 25), 
                        (48, 15), (24, 10), (56, 20), (32, 15), (16, 15),
                        (8, 10),  (24, 15), (8, 15),  (32, 25), (8, 5)]

            test_domain_task = [(32, 20), (32, 5), (24, 20), (16, 25), (24, 5), 
                                (16, 10), (40, 20), (8, 25), (24, 25), (40, 10)]


        elif symmetry == "Rotation":
            train_task = [(27, 2), (33, 0), (3, 2), (28, 3),(9, 0),
                        (12, 3), (22, 1), (8, 3), (30, 1), (25, 0),
                        (16, 3), (11, 2), (23, 2), (29, 0), (36, 3),
                        (26, 1), (1, 0), (35, 2), (19, 2), (34, 1),
                        (4, 3), (2, 1), (7, 2), (31, 2), (17, 0)]

            test_domain_task = [(6, 1), (14, 1), (15, 2), (10, 1), (18, 1),
                                (20, 3), (24, 3), (13, 0), (21, 0), (5, 0)]

        elif symmetry == "Scale":
            train_task = [27,  9,  7, 11,  4, 26, 35,
                        2, 29, 10, 34, 12, 37, 28,
                        18, 24,  8, 14, 1, 31, 25,
                        0, 19, 15, 36,  3, 20, 13]

            test_domain_task = [ 5, 30, 16, 23, 33,
                                6, 17, 22, 21, 32]
        else:
            print("Invalid dataset name entered!")
            
        valid_task = train_task
        test_future_task = train_task

        return train_task,train_time,valid_task,valid_time,test_domain_task,test_domain_time,test_future_task,test_future_time


    def prepare_data(self) -> None:
        Wang2022Dataset.download_and_extract(root_dir=self.hparams.root_dir, direc=self.hparams.symmetry)

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
            direc = Wang2022Dataset.download_and_extract(
                root_dir=self.hparams.root_dir, logger=self.logger, direc=self.hparams.symmetry
            )

            params = {
                'input_length': self.hparams.input_length,
                'mid': self.hparams.mid,
                'direc': direc,
                'stack': True
            }

            self.data_train = Wang2022Dataset(**params, sample_list=self.train_time, task_list=self.train_task, output_length=self.hparams.output_length)
            self.data_val = Wang2022Dataset(**params, sample_list=self.val_time, task_list=self.val_task, output_length=self.hparams.output_length)
            if self.hparams.future:
                self.data_test = Wang2022Dataset(**params, sample_list=self.test_future_time, task_list=self.test_future_task, output_length=20)
            else: 
                self.data_test = Wang2022Dataset(**params, sample_list=self.test_domain_time, task_list=self.test_domain_task, output_length=20)

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
    dm = Wang2022Datamodule(
        symmetry="Translation",
    )
    dm.prepare_data()
