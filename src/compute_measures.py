from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.metrics.equivariance_error import get_equivariance_error
from src.metrics.lie_derivative import get_lie_equiv_err
from src.metrics.sharpness import get_sharpness
# from src.metrics.hessian_spectrum import get_spectrum
from src.utils.wandb import download_config_file, get_model_and_data_modules_from_config
from collections import defaultdict
import os
import numpy as np

import wandb
import json
# before running this script please log in to wandb, like wandb.login(key=userdata.get("wandb_key"))

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. from src import utils)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set root_dir to "." in "configs/paths/default.yaml"
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

def get_checkpoint_dict(path_dict, run):
    
    checkpoint_dict = {}
    
    for name in path_dict.keys():
        current_checkpoint = run.use_artifact(path_dict[name], type="model")
        current_dir = current_checkpoint.download()
        checkpoint_dict[name] = torch.load(current_dir + "/model.ckpt")

    return checkpoint_dict


def parse_ckpt_path(ckpt_path: str):

    parts = ckpt_path.split('/')
    entity = parts[0]  # The first part is the organization
    project = parts[1]  # The second part is the project
    run_id_parts = parts[2].split(':')  # Split the last part to separate model ID and version
    run_id = run_id_parts[0].replace('model-', '')  # Remove 'model-' prefix and get the model ID
    
    return entity, project, run_id

def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, dict):
        # Recursively convert dictionary items
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

        
@hydra.main(version_base="1.3", config_path="../configs", config_name="compute_measures.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    run = wandb.init()

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    checkpoint_dict = get_checkpoint_dict(path_dict = cfg.ckpt_path_dict, run=run)

    metric_dict = defaultdict(dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in checkpoint_dict.keys():
        log.info('obtaining spectrum for checkpoint', name)

        #PARSE THE CKPT_PATH_DICT
        entity, project, run_id = parse_ckpt_path(cfg.ckpt_path_dict[name])
        
        #GET THE MODEL AND DATAMODULE
        config = download_config_file(entity, project, run_id)
        model, datamodule = get_model_and_data_modules_from_config(config)

        #Do what lightning would normally do
        model.to(device)
        datamodule.setup()
        

        #COMPUTE THE WANTED METRICS
        model.load_state_dict(checkpoint_dict[name]["state_dict"])
        if cfg.get("equivariance_error"):
           metric_dict[name]["equivariance_error"] = convert_to_serializable(get_equivariance_error(model, datamodule, device))
        if cfg.get("lie_derivative"):
          metric_dict[name]["lie_derivative"] = convert_to_serializable(get_lie_equiv_err(model, datamodule, device))
        if cfg.get("sharpness"):
          metric_dict[name]["sharpness"] = convert_to_serializable(get_sharpness(model, datamodule, device))
        # if cfg.get("spectrum"):
        #   metric_dict[name]["spectrum"] = convert_to_serializable(get_spectrum(cfg, config, datamodule, model))
    
        storage_path = cfg.storage_location + "/metrics.json"
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Write the metrics to a file
        with open(storage_path, "w") as outfile: 
            json.dump(metric_dict, outfile)

        #also put the file on wandb

    run.finish()

if __name__ == "_main_":
    main()