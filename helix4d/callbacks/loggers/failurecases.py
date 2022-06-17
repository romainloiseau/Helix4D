from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .base import BaseLogger

class FailureCasesLogger(BaseLogger):    

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.failure_cases = {
            "loss": torch.tensor(self.hparams.N * [0]).float(),
            "batch": self.hparams.N * [[]],
            "pred": self.hparams.N * [[]],
            "scanid": torch.tensor(self.hparams.N * [-10]).float()
        }

    @torch.no_grad()
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.do_greedy_step(pl_module.current_epoch):
            if (torch.abs(self.failure_cases["scanid"] - batch.scanid[0].detach().cpu()).min() > 100) and outputs["loss"] > self.failure_cases["loss"].min():
                idx = self.failure_cases["loss"].argmin()
                self.failure_cases["loss"][idx] = outputs["loss"].detach().cpu()
                self.failure_cases["batch"][idx] = batch.cpu()
                self.failure_cases["pred"][idx] = outputs["point_pred"].detach().cpu()
                self.failure_cases["scanid"][idx] = batch.scanid[0].detach().cpu()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.do_greedy_step(pl_module.current_epoch):
            for i, idx in enumerate(self.failure_cases["loss"].argsort()):
                if self.failure_cases["batch"][idx] != []:
                    self.greedy_images(
                        pl_module.logger.experiment, pl_module.current_epoch,
                        self.failure_cases["batch"][idx], f"val_failurecases/{i}", self.failure_cases["pred"][idx]
                    )