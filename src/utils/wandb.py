import wandb
from wandb.apis.public import Run

def get_all_checkpoints(run_id: str, project: str, entity: str) -> list[str]:
    """Get all checkpoints from a run.

    :param run_id: The run id.
    :param project: The project name.
    :param entity: The entity name.
    :return: A list of checkpoint names ["model-runid:v0", ...].
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return [a.name for a in run.logged_artifacts() if a.type == "model"]


def download_artifact(run: Run, artifact_name: str, project: str, entity: str) -> str:
    """ Download artifact from wandb, link it to current run and return the path. 
    :param run: The current run. Wandb will automatically link the artifact to this run.
    :param artifact_name: The artifact name.
    :param project: The project name.
    """
    artifact = run.use_artifact(f"{entity}/{project}/{artifact_name}", type='model')
    return artifact.download()
