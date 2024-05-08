from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from neuralyze import get_hessian_max_spectrum
import wandb
import json
# before running this script please log in to wandb, like wandb.login(key=userdata.get("wandb_key"))

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def get_checkpoint_dict(path_dict):
    
    checkpoint_dict = {}
    run = wandb.init()

    for name in path_dict.keys():
        current_checkpoint = run.use_artifact(path_dict[name], type="model")
        current_dir = current_checkpoint.download()
        checkpoint_dict[name] = torch.load(current_dir + "/model.ckpt")

    run.finish()

    return checkpoint_dict

@task_wrapper
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

        
@hydra.main(version_base="1.3", config_path="../configs", config_name="hessian_spectra.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    #load the model 
    ckpt_path_dict = json.loads(cfg.ckpt_path_dict)

    checkpoint_dict = get_checkpoint_dict(path = ckpt_path_dict)
    spectrum_dict = {}

    for name in checkpoint_dict.keys():
        print('obtaining spectrum for checkpoint', name)
        model.load_state_dict(checkpoint_dict[name]["state_dict"])
        spectrum_dict[name] = get_spectrum(cfg, datamodule, model)
    
    with open(cfg.storage_location + "/spectra.json", "w") as outfile: 
        json.dump(spectrum_dict, outfile)

if __name__ == "__main__":
    main()