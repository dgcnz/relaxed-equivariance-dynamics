from neuralyze import get_hessian_max_spectrum
import torch
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Tuple

def get_spectrum(cfg: DictConfig, config, datamodule, model) -> List:
    
    #get the dataset from the datamodule 
    dataset = datamodule.data_train

    # get criterion (might have to make this selectable in the future)
    loss_fn = torch.nn.MSELoss()
    if cfg.get("wang2022_loss"):
        loss_fn = summed_mse_loss(cfg, model)
    else:
        ValueError("No loss function specified")

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

def summed_mse_loss(cfg, model):
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with_weight_constraint = hasattr(model.net, "get_weight_constraint")
    criterion = torch.nn.MSELoss()
    
    def loss_fn(xx, yy): #input, target
        mse = torch.tensor(0.0, device=device)
        
        for y in yy.transpose(0, 1):
            im = model.forward(xx)
            xx = torch.cat([xx[:, im.shape[1] :], im], 1)
            mse += criterion(im, y)

        if with_weight_constraint:
            weight_constraint = criterion(model.net.get_weight_constraint(), torch.tensor(0).float().cuda())
            
            loss = mse + weight_constraint
        else:
            weight_constraint = None
            loss = mse

        return loss
    
    return loss_fn