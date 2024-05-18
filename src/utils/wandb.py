import wandb
from wandb.apis.public import Run
from omegaconf import OmegaConf, DictConfig
import hydra
from lightning import LightningModule, LightningDataModule


def get_all_checkpoints(run_id: str, project: str, entity: str) -> list[str]:
    """Get all checkpoints from a run.

    :param run_id: The run id (e.g. rgo48mzm). Not to be confused with the run name (e.g. crimson-valley-31).
    :param project: The project name.
    :param entity: The entity name.
    :return: A list of checkpoint names ["model-runid:v0", ...].
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return [a.name for a in run.logged_artifacts() if a.type == "model"]


def download_artifact(run: Run, artifact_name: str, project: str, entity: str) -> str:
    """Download artifact from wandb, link it to current run and return the path.
    :param run: The current run. Wandb will automatically link the artifact to this run.
    :param artifact_name: The artifact name.
    :param project: The project name.
    """
    artifact = run.use_artifact(f"{entity}/{project}/{artifact_name}", type="model")
    return artifact.download()


def download_config_file(entity: str, project: str, run_id: str) -> DictConfig:
    """Download a config file from wandb and return the path.
    :param entity: The entity name.
    :param project: The project name.
    :param run_id: The run id.

    :return: The config file as a DictConfig. Keys: ["model", "data", "trainer", "callbacks", "tags", etc.]
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    file = run.file("files/config.yaml")
    fp = file.download(replace=True)
    conf = OmegaConf.load(fp)
    return conf


def get_model_and_data_modules_from_config(
    wandb_config: DictConfig,
) -> tuple[LightningModule, LightningDataModule]:
    """
    Instantiates the model and datamodule from a config file downloaded from wandb.
    :param wandb_config: The config file downloaded by `download_config_file`.
    """
    model = hydra.utils.instantiate(wandb_config.model.value)
    datamodule = hydra.utils.instantiate(wandb_config.data.value, root_dir="data")
    return model, datamodule
