
from dataclasses import dataclass
from pydoc import resolve
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

import os, logging
import torch
import torchvision

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    # logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # logger.info(cfg.dataloaders)
    # logger.info(cfg.optimizer)

    # https://hydra.cc/docs/advanced/instantiate_objects/overview/
    model = hydra.utils.instantiate(cfg.models)
    print(model)


if __name__ == "__main__":
    # this is a DictConfig object: https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#creating
    # dataloader_config = OmegaConf.load("./configs/dataloaders/mnist_dataloaders.yaml")
    # print(dataloader_config)
    train()  # unique run in outputs folder

