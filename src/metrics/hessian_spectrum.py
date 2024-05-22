from neuralyze import get_hessian_max_spectrum
import torch
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple

from src.pyhessian import hessian
from torch.utils.data import DataLoader, IterableDataset, Dataset as MapDataset, Subset
from tqdm import tqdm

def get_spectrum(cfg: DictConfig, config, datamodule, model) -> List:
    
    #get the dataset from the datamodule 
    dataset = datamodule.data_train

    # get criterion (might have to make this selectable in the future)
    loss_fn = torch.nn.CrossEntropyLoss()

    spectrum = get_hessian_max_spectrum(
        model=model,
        criterion=loss_fn,
        train_dataset= dataset,
        batch_size = config.data.batch_size,
        percentage_data = cfg.percentage_data,
        weight_decay = config.model.optimizer.weight_decay,
        hessian_top_k= cfg.top_k,
        hessian_tol = cfg.tol,
        hessian_max_iter= cfg.max_iter,
        cuda = cfg.cuda,
        verbose = cfg.verbose,
    )

    return spectrum



### Im sorry diego but I cant really use the package because I had to fork pyhessian
def get_hessian_max_spectrum(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_dataset: MapDataset | IterableDataset,
    batch_size: int = 16,
    percentage_data: float | None = None,
    weight_decay: Optional[float] = None,
    hessian_top_k: int = 5,
    hessian_tol: float = 1e-3,
    hessian_max_iter: int = 1000,
    cuda: bool = False,
    verbose: bool = True,
) -> list[float]:
    """
    Compute batched top-k hessian max eigenvalues of a model with respect to a criterion and a dataloader
    using [power-iteration](https://en.wikipedia.org/wiki/Power_iteration) as per
    [Park et al. 2021](https://arxiv.org/abs/2105.12639) and
    [Park et al. 2022](https://github.com/xxxnell/how-do-vits-work/issues/12#issuecomment-1113991017)

    For reference, the params outlined in [Park et al. 2022] are:
    - batch_size: 16
    - percentage_data: 0.1

    :param model: Neural network model
    :param criterion: Loss function
    :param train_dataset: Dataset used for training
    :param batch_size: Batch size used in training
    :param percentage_data: Percentage of data to use from the dataset (randomly sampled)
    :param weight_decay: L2 regularization strength used in training
    :param hessian_top_k: Number of top eigenvalues to compute
    :param hessian_tol: Tolerance for eigenvalue computation
    :param hessian_max_iter: Maximum number of iterations for eigenvalue computation
    :param cuda: Whether to use GPU
    :param verbose: Whether to display progress bar
    """
    if isinstance(train_dataset, IterableDataset):
        raise ValueError("IterableDataset not supported. Use a Dataset that supports __len__.")
    
    if percentage_data is not None:
        num_samples = int(len(train_dataset) * percentage_data)
        train_dataset = Subset(train_dataset, torch.randperm(len(train_dataset))[:num_samples])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    max_eigens: list[float] = []
    for xs, ys in tqdm(train_dataloader, disable=not verbose):
        hessian_comp = hessian(
            model, criterion, data=(xs, ys), cuda=cuda
        )
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=hessian_top_k, tol=hessian_tol, maxIter=hessian_max_iter)
        max_eigens.extend(top_eigenvalues)
    return max_eigens