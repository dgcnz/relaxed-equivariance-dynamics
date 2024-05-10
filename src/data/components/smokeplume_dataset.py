from src.data.components.wang2022_dataset import Wang2022Dataset
import requests
import zipfile
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)


class SmokePlumeDataset(Wang2022Dataset):
    ETURL = "https://huggingface.co/datasets/dl2-g32/smokeplume/resolve/main/equivariance_test.zip?download=true"
    data_dir: Path

    def __init__(
        self,
        root: str,
        equivariance_level: int,
        sample_list: list[int],
        input_length: int = 1,
        mid: int = 3,
        output_length: int = 6,
        task_list: list[int] = [0, 1, 2, 3],
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        root = Path(root)

        direc = SmokePlumeDataset.download_and_extract(
            root_dir=root, logger=self.logger
        )
        direc = direc / f"E_{equivariance_level}"

        super().__init__(
            input_length=input_length,
            mid=mid,
            output_length=output_length,
            direc=str(direc),
            task_list=task_list,
            sample_list=sample_list,
            stack=True
        )

    @classmethod
    def get_data_dir(cls, root_dir: Path):
        data_dir = root_dir / "smokeplume"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @classmethod
    def download_and_extract(
        cls, root_dir: str, logger: Optional[logging.Logger] = None
    ):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
        root_dir = Path(root_dir)
        data_dir = cls.get_data_dir(root_dir)
        out_folder = data_dir / "equivariance_test"
        if out_folder.exists():
            logger.info(f"Data already downloaded and extracted at {out_folder}")
            return out_folder

        logger.info(f"Downloading and extracting data to {out_folder}")
        r = requests.get(SmokePlumeDataset.ETURL, allow_redirects=True)
        with open(data_dir / "equivariance_test.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(data_dir / "equivariance_test.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)
        # remove the zip file
        (data_dir / "equivariance_test.zip").unlink()
        return out_folder


if __name__ == "__main__":
    out_length = 6
    input_length = 1
    mid = input_length + 2
    train_time = list(range(0, 30))
    level = 0
    data_direc = "notebooks/figure_4_data/E_" + str(level)
    train_task = [0, 1, 2, 3]
    train_set = SmokePlumeDataset(
        root="data",
        input_length=input_length,
        mid=mid,
        output_length=out_length,
        task_list=train_task,
        sample_list=train_time,
        equivariance_level=level,
    )
    x, y = train_set[0]
    print(x.shape)
    print(y.shape)
