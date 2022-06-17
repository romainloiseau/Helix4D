import torch
import spconv.pytorch as spconv

import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from helix4d.trainers import get_trainer

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    if cfg.seed != 0:
        pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )
    
    if "helixnet" in cfg.data._target_:
        cfg.data._target_ = f"HelixNet.{cfg.data._target_}"
    datamodule = hydra.utils.instantiate(cfg.data)

    trainer = get_trainer(cfg)

    if cfg.model.load_weights != "":
        print(f"Loading weights from '{cfg.model.load_weights}'")
        for gpu in cfg.trainer.gpus:
            model.load_state_dict(torch.load(cfg.model.load_weights, map_location=f"cuda:{gpu}")['state_dict'])

    getattr(trainer, cfg.mode)(model, datamodule=datamodule)

if __name__ == '__main__':
    main()