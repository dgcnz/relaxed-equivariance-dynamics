from typing import Optional

import hydra
import rootutils
from lightning import LightningModule
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    log.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )


if __name__ == "__main__":
    main()
