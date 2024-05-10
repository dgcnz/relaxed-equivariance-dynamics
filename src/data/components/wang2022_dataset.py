import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Union



class Wang2022Dataset(TorchDataset):
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
        self.direc = direc
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
        if not self.stack:
            x = self.data_lists[task_idx][
                (self.mid - self.input_length + self.sample_list[sample_idx]) : (
                    self.mid + self.sample_list[sample_idx]
                )
            ]
        else:
            x = self.data_lists[task_idx][
                (self.mid - self.input_length + self.sample_list[sample_idx]) : (
                    self.mid + self.sample_list[sample_idx]
                )
            ].reshape(-1, y.shape[-2], y.shape[-1])
        return x.float(), y.float()

