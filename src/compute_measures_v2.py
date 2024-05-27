import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig
from src.metrics.equivariance_error import get_equivariance_error
from src.metrics.lie_derivative import get_lie_equiv_err
from src.metrics.sharpness import get_sharpness
from src.metrics.hessian_spectrum import get_spectrum
from src.utils.wandb import download_config_file, get_model_and_data_modules_from_config
from collections import defaultdict
import os
import numpy as np
from pathlib import Path

import wandb
import json

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.wandb import (
    download_config_file,
    download_artifact,
    get_model_and_data_modules_from_config,
)

log = RankedLogger(__name__, rank_zero_only=True)


def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, dict):
        # Recursively convert dictionary items
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="compute_measures_v2.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for training.
    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    # !python -m src.compute_measures ++weight_decay=1e-5 percentage_data=20 'ckpt_path_dict={path_dict_best}' spectrum=True sharpness=True
    entity, project, artifact_name = cfg.ckpt_path.split("/")
    run_id = artifact_name.split(":")[0].split("-")[1]
    with wandb.init(entity=entity, project=project) as run:
        artifact_dir = download_artifact(run, artifact_name, project, entity)
        config = download_config_file(entity, project, run_id)
        model, datamodule = get_model_and_data_modules_from_config(config)
        model = model.__class__.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
        log.info("obtaining spectrum for checkpoint", cfg.ckpt_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        datamodule.setup()

        metric_dict = dict()
        if cfg.get("equivariance_error"):
            metric_dict["equivariance_error"] = convert_to_serializable(
                get_equivariance_error(model, datamodule, device)
            )
        if cfg.get("lie_derivative"):
            metric_dict["lie_derivative"] = convert_to_serializable(
                get_lie_equiv_err(model, datamodule, device)
            )
        if cfg.get("sharpness"):
            metric_dict["sharpness"] = convert_to_serializable(
                get_sharpness(model, datamodule, device)
            ).item()
            run.summary["sharpness"] = metric_dict["sharpness"]
        if cfg.get("spectrum"):
            metric_dict["spectrum"] = convert_to_serializable(
                get_spectrum(cfg, config, datamodule, model)
            )
            run.summary["max_eigenvalue"] = np.max(metric_dict["spectrum"])
            run.summary["min_eigenvalue"] = np.min(metric_dict["spectrum"])
            run.summary["mean_eigenvalue"] = np.mean(metric_dict["spectrum"])
            try:
                data = [[s] for s in metric_dict["spectrum"]]
                table = wandb.Table(data=data, columns=["eigenvalue_mag"])
                run.log(
                    {
                        "spectrum": wandb.plot.histogram(
                            table,
                            "eigenvalue_mag",
                            title="Max Hessian Spectrum",
                        )
                    }
                )
            except:
                print("Failed to log spectrum")
                pass

            # log all eigenvalues

        storage_path = cfg.storage_location + "/metrics.json"
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Write the metrics to a file
        with open(storage_path, "w") as outfile:
            json.dump(metric_dict, outfile)
        # also put the file on wandb as an artifact
        metrics_artifact = wandb.Artifact("metrics", type="metrics")
        metrics_artifact.add_file(storage_path)
        run.log_artifact(metrics_artifact)


if __name__ == "__main__":
    main()
