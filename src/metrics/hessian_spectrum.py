from neuralyze import get_hessian_max_spectrum
import torch
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple

def get_spectrum(cfg: DictConfig, datamodule, model) -> List:
    
    #get the dataset from the datamodule 
    dataset = datamodule.data_train

    # get criterion (might have to make this selectable in the future)
    loss_fn = torch.nn.CrossEntropyLoss()

    #weight_decay = 1e-5

    spectrum = get_hessian_max_spectrum(
        model=model,
        criterion=loss_fn,
        train_dataset= dataset,
        batch_size = cfg.batch_size,
        percentage_data = cfg.percentage_data,
        weight_decay = cfg.weight_decay,
        hessian_top_k= cfg.top_k,
        hessian_tol = cfg.tol,
        hessian_max_iter= cfg.max_iter,
        cuda = cfg.cuda,
        verbose = cfg.verbose,
    )

    return spectrum