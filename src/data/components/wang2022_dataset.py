import requests
import zipfile
from pathlib import Path
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Union


class Wang2022Dataset(TorchDataset):
    ETURL_dict = {"Translation": "https://huggingface.co/datasets/dl2-g32/smokeplume/resolve/main/Smoke/Translation.zip?download=true ",
                  "Scale": "https://huggingface.co/datasets/dl2-g32/smokeplume/resolve/main/Smoke/Scale.zip?download=true ",
                  "Rotation": "https://huggingface.co/datasets/dl2-g32/smokeplume/resolve/main/Smoke/Rotation.zip?download=true "}
    data_dir: Path

    def __init__(
        self,
        input_length: int,
        mid: int,
        output_length: int,
        direc: str,
        task_list: Union[list[int], list[tuple[int, int]]],
        sample_list: list[int],
        stack: bool = False,
    ):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = str(direc)
        self.task_list = task_list
        self.sample_list = sample_list
        self.stack = stack
        if isinstance(task_list[0], tuple):
            assert len(task_list[0]) == 2
            self.data_lists = [
                torch.load(
                    self.direc + "/raw_data_" + str(idx[0]) + "_" + str(idx[1]) + ".pt"
                )
                for idx in task_list
            ]
        elif isinstance(task_list[0], int):
            self.data_lists = [
                torch.load(self.direc + "/raw_data_" + str(idx) + ".pt")
                for idx in task_list
            ]
        else:
            raise ValueError("task_list should be a list of int or tuple of int")

    def __len__(self):
        return len(self.task_list) * len(self.sample_list)

    def __getitem__(self, index: int):
        task_idx = index // len(self.sample_list)
        sample_idx = index % len(self.sample_list)
        y = self.data_lists[task_idx][
            (self.sample_list[sample_idx] + self.mid) : (
                self.sample_list[sample_idx] + self.mid + self.output_length
            )
        ]
        x = self.data_lists[task_idx][
            (self.mid - self.input_length + self.sample_list[sample_idx]) : (
                self.mid + self.sample_list[sample_idx]
            )
        ]
        if self.stack:
            # x: (input_length, channels, H, W)
            # Comments:
            #   I think channels can be interpreted here as the dimensionality of the vector
            #   For smokeplume we are tracking velocity so v = [dx/dt, dy/dt]
            #   Thus x: (1, 2, 64, 64)
            x = x.reshape(-1, y.shape[-2], y.shape[-1])
            # x: (input_length * channels, H, W)
        return x.float(), y.float()
    
    @classmethod
    def get_data_dir(cls, root_dir: Path):
        data_dir = root_dir / "wang2022relaxed"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @classmethod
    def download_and_extract(
        cls, root_dir: str, direc: str, logger: Optional[logging.Logger] = None 
    ):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
        root_dir = Path(root_dir)
        data_dir = cls.get_data_dir(root_dir)

        out_folder = data_dir / direc
        if out_folder.exists():
            logger.info(f"Data already downloaded and extracted at {out_folder}")
            return out_folder

        logger.info(f"Downloading and extracting data to {out_folder}")
        r = requests.get(Wang2022Dataset.ETURL_dict[direc], allow_redirects=True)
        with open(data_dir / "equivariance_test.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(data_dir / "equivariance_test.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)
        # remove the zip file
        (data_dir / "equivariance_test.zip").unlink()
        return out_folder
