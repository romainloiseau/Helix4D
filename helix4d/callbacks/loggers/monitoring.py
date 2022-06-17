from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .base import BaseLogger

class MonitoringLogger(BaseLogger):

    def log_step(self, pl_module):
        pl_module.log(f'step', 1 + pl_module.current_epoch, on_step=False, on_epoch=True)

    def greedy_step(self, trainer, pl_module, batch, batch_idx, tag, outputs):
        if batch_idx == 0 and self.do_greedy_step(pl_module.current_epoch):
            if tag == "train":
                self.greedy_histograms(trainer, pl_module, pl_module.current_epoch, batch)
            self.greedy_images(
                pl_module.logger.experiment, pl_module.current_epoch, batch, tag,
                outputs["point_pred"],
                assignments=outputs["point2voxel"] if "point2voxel" in outputs.keys() else None,
                assignments_maps=outputs["assignments_maps"] if "assignments_maps" in outputs.keys() else None
            )

    def greedy_histograms(self, trainer, pl_module, current_epoch, batch):
        features = batch.features.detach().cpu()
        pos = batch.pos.detach().cpu()

        features_names = trainer.datamodule.get_features_names()
        for i, kk in enumerate(features_names):
            if i < batch.features.size(-1):
                pl_module.logger.experiment.add_histogram(
                    f"features/{kk}",
                    features[:, i].flatten(),
                    global_step=current_epoch
                )
        for i, kk in enumerate(["x", "y", "z"]):
            pl_module.logger.experiment.add_histogram(
                f"positions/{kk}",
                pos[:, i].flatten(),
                global_step=current_epoch
            )
        
        weights_first = pl_module.point_encoder[0].weight.data.detach().cpu()
        for i in range(weights_first.size(-1)):
            pl_module.logger.experiment.add_histogram(
                f"first_layer_weights/{features_names[i] if i < len(features_names) else f'label_{i - len(features_names) + 1}'}",
                weights_first[:, i].flatten(),
                global_step=current_epoch
            )

    @torch.no_grad()
    def log_metrics(self, trainer, pl_module: "pl.LightningModule", tag, batch, batch_idx, outputs):
        with torch.profiler.record_function(f"LOGGERS"):
            
            getattr(pl_module, f'IoU_{tag}').update(outputs["point_pred"], batch.point_y)

            pl_module.log(f'Loss_total/{tag}',
                outputs["loss"], on_step=False, on_epoch=True, batch_size=outputs["batch_size"])
            pl_module.log(f'Loss/point_crossentropy/{tag}',
                outputs["point_crossentropy"], on_step=False, on_epoch=True, batch_size=outputs["batch_size"])
            pl_module.log(f'Loss/point_lovasz/{tag}',
                outputs["point_lovasz"], on_step=False, on_epoch=True, batch_size=outputs["batch_size"])

            self.greedy_step(
                trainer, pl_module,
                batch, batch_idx, tag,
                outputs
            )

    def log_ious(self, pl_module, tag):
        curr_iou = getattr(pl_module, f"IoU_{tag}").compute().detach().cpu().numpy()
        if curr_iou.sum() != 0:
            pl_module.log(f'IoU/{tag}', curr_iou.mean(), on_step=False, on_epoch=True)

            if self.do_greedy_step(pl_module.current_epoch):
                with torch.profiler.record_function(f"CONFMAT"):
                    pl_module.logger.experiment.add_image(
                        f"confmat/{tag}", self.image_confusion_matrix(
                            getattr(pl_module, f"IoU_{tag}").confmat.detach().cpu().numpy(), curr_iou),
                        global_step=pl_module.current_epoch, dataformats='HWC'
                    )
            getattr(pl_module, f"IoU_{tag}").reset()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        self.log_metrics(trainer, pl_module, "train", batch, batch_idx, outputs)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.log_metrics(trainer, pl_module, "val", batch, batch_idx, outputs)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.log_metrics(trainer, pl_module, "test", batch, batch_idx, outputs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_ious(pl_module, "train")
        self.log_step(pl_module)

        for param_group in pl_module.optimizers().param_groups:
            pl_module.log(f'epoch/lr/{param_group["name"]}',
                     float(param_group["lr"]), on_step=False, on_epoch=True)
            pl_module.log(f'epoch/wd/{param_group["name"]}',
                     float(param_group["weight_decay"]), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_ious(pl_module, "val")
        self.log_step(pl_module)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_ious(pl_module, "test")
        self.log_step(pl_module)