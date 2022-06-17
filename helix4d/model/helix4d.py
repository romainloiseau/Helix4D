import math
import time

import numpy as np
import pytorch_lightning as pl
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch_scatter
import torchmetrics
from torch import nn

from HelixNet.helixnet.laserscan import HelixNetThetaCorrector

import helix4d.model.voxel.conv.utils_conv as ConvZoo
from helix4d.metrics.lovasz_losses import LovaszLoss
from helix4d.model import voxel as V
from helix4d.utils import apply_learning_map

from .logging import LoggingModel
from .online import OnlineModel
from .point import decoders as PD
from .point import encoders as PE

from torch_geometric.data import Data

@torch.jit.script
def from_grid_to_spconvgrid(grid: torch.Tensor, batch_size: int):
    grid = grid.to(torch.int32)
    return torch.cat([(grid[:, 0] + batch_size*grid[:, 1]).unsqueeze(-1), grid[:, 2:]], -1).contiguous()

@torch.jit.script
def retrieve_pos_and_maps(voxel_pos, batch_size: int, spconvgrid, slice_indices, from_slicegrid_to_sliceindices, full_split_hierarchy):
    down_spconvgrid = torch.div(spconvgrid, full_split_hierarchy, rounding_mode="floor")

    down_spconvindices = (down_spconvgrid * from_slicegrid_to_sliceindices).sum(-1)

    spconvtensor_indices = (slice_indices * from_slicegrid_to_sliceindices).sum(-1)

    voxel_pos = torch_scatter.scatter_mean(voxel_pos, down_spconvindices, dim=0)[spconvtensor_indices]

    slice_indices = torch.cat([
                (slice_indices[:, 0] % batch_size).unsqueeze(-1),
                (torch.div(slice_indices[:, 0], batch_size, rounding_mode="floor")).unsqueeze(-1),
                slice_indices[:, 1:]]
                , -1)
            
    return voxel_pos,down_spconvindices,slice_indices

@torch.jit.script
def compute_point2voxel_map(highres_batch, highres_voxelind):
    highres_ind = torch.cat(
        [highres_batch.unsqueeze(-1), highres_voxelind], -1
        )
        
    lowres_ind, highres2lowres = torch.unique(
        highres_ind, dim=0, return_inverse=True)
    return lowres_ind, highres2lowres

@torch.jit.script
def voxelize(highres_features, highres_pos, highres2lowres):
    return torch_scatter.scatter_max(highres_features, highres2lowres, 0)[0], torch_scatter.scatter_mean(highres_pos, highres2lowres, 0)

@torch.jit.script
def upsample(point2voxel, point_features, voxel_features):
        return torch.cat([
            point_features,
            torch.index_select(voxel_features, 0, point2voxel)
        ], -1)

@torch.jit.script
def upsample_backprop( backprop, point2voxel, point_features, voxel_features):
    return torch.cat([
        point_features[backprop],
        torch.index_select(voxel_features, 0, point2voxel[backprop])
    ], -1)

class Helix4D(pl.LightningModule, LoggingModel, OnlineModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.__initmodel__()
        self.__initmetrics__()
        self.__compute_nparams__()

    def __compute_nparams__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def __initmetrics__(self):
        self.IoU_train = torchmetrics.IoU(
            self.hparams.data.output_dim, 0, reduction="none")
        self.IoU_val = torchmetrics.IoU(
            self.hparams.data.output_dim, 0, reduction="none")
        self.IoU_test = torchmetrics.IoU(
            self.hparams.data.output_dim, 0, reduction="none")

        self.criterion_lovasz = LovaszLoss(
            n_classes=self.hparams.data.output_dim,
            ignore_index=0
        )
        
        self.criterion_crossentropy = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=0,
        )

    def __initmodel__(self):
        self.__init_pointencoder__()
        self.__init_voxelmodel__()

        self.point_decoder = PD.LinearDecoder(
            dim_in=self.hparams.point_encoder[-1] +
            self.hparams.voxel_encoder,
            decoder=self.hparams.point_decoder,
            dim_out=self.hparams.data.output_dim
        )     

    def __init_voxelmodel__(self):
        
        dim_in, dim_out = self.hparams.point_encoder[-1], self.hparams.voxel_encoder

        self.point2voxel_linear = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU()
        )

        self.__initspconvvoxelshape__()

        for i in range(self.hparams.data.n_hierarchy):
            dim_in, dim_out = dim_out, self.hparams.voxel_encoder_mul[len(self.hparams.voxel_encoder_mul) - self.hparams.data.n_hierarchy + i]*dim_out

            setattr(self, f"H{i}_before",
                nn.Sequential(*[getattr(ConvZoo, self.hparams.voxel_function)(dim_in, dim_in, key=f"{i}") for _ in range(self.hparams.n_voxel_layers)])
            )

            if self.hparams.pooling == "learned":
                setattr(self, f"H{i}_H{i+1}",
                    ConvZoo.MySparseConv3d(
                        dim_in, dim_out,
                        kernel_size=tuple(self.hparams.data.split_hierarchy[i]),
                        stride=tuple(self.hparams.data.split_hierarchy[i]),
                        indice_key=f"{i}_{i+1}_learnedpool",
                    )
                )
                setattr(self, f"H{i}_H{i+1}_norm", nn.Sequential(nn.LeakyReLU(), nn.BatchNorm1d(dim_out)))

            elif self.hparams.pooling == "max":
                setattr(self, f"H{i}_H{i+1}",
                    spconv.SparseMaxPool3d(
                        kernel_size=tuple(self.hparams.data.split_hierarchy[i]),
                        stride=tuple(self.hparams.data.split_hierarchy[i]),
                        indice_key=f"{i}_{i+1}_learnedpool", 
                    )
                )
                setattr(self, f"H{i}_H{i+1}_norm", nn.Sequential(nn.Linear(dim_in, dim_out), nn.LeakyReLU(), nn.BatchNorm1d(dim_out)))

            else:
                raise NotImplementedError

            setattr(self, f"H{i+1}_H{i}",
                    ConvZoo.MySparseInverseConv3d(
                        dim_out, dim_in,
                        kernel_size=tuple(self.hparams.data.split_hierarchy[i]),
                        indice_key=f"{i}_{i+1}_learnedpool",
                    )
                )

            setattr(self, f"H{i}_after",
                nn.Sequential(*[getattr(ConvZoo, self.hparams.voxel_function)(dim_in, dim_in, key=f"{i}") for _ in range(self.hparams.n_voxel_layers)])
            )

        self.hparams.transformer.relative_positional_ntime = 1 + len(self.hparams.data.temporal_scan_sequence) * self.hparams.transformer.temporal_buckets
        self.voxel_transformer = V.Transformer(
            dim_in=dim_out,
            transformer=self.hparams.transformer,
            temporal_transformer=self.hparams.data.temporal_transformer,
            slices_per_rotation = self.hparams.data.slices_per_rotation
        )


    def __initspconvvoxelshape__(self):
        voxel_res = np.array(self.hparams.data.voxel_res)
        voxel_res[1] = math.pi * voxel_res[1] / 180
        full_split_hierarchy = np.prod(np.array(self.hparams.data.split_hierarchy)[:self.hparams.data.n_hierarchy], 0)
        thin_voxel_res = voxel_res / full_split_hierarchy

        self.register_buffer("full_split_hierarchy", torch.tensor([1] + list(full_split_hierarchy)))

        voxelind_max_r = int(self.hparams.data.polar_max_r / thin_voxel_res[0]) + 1
        voxelind_max_theta = math.ceil(
            2*math.pi / (self.hparams.data.slices_per_rotation * thin_voxel_res[1]))
        voxelind_max_z = int(
            (self.hparams.data.polar_max_z - self.hparams.data.polar_min_z) / thin_voxel_res[2]) + 1

        if self.hparams.data.name == "helixnet":
            max_delta_rad = (HelixNetThetaCorrector()._MAX_DIFF_)/2.
            add_slice_max_theta = math.ceil(max_delta_rad/voxel_res[1])
            max_delta_rad = int((voxel_res[1] / thin_voxel_res[1]) * add_slice_max_theta)
            voxelind_max_theta += 2*max_delta_rad

        self.voxel_shape = [voxelind_max_r, voxelind_max_theta, voxelind_max_z]
        self.avail_slices = self.hparams.data.slices_per_rotation*(1+len(self.hparams.data.temporal_scan_sequence))

        slice_max_r = int(self.hparams.data.polar_max_r / voxel_res[0]) + 1
        slice_max_theta = math.ceil(
            2*math.pi / (self.hparams.data.slices_per_rotation * voxel_res[1]))

        if self.hparams.data.name == "helixnet":
            slice_max_theta += 2*add_slice_max_theta

        slice_max_z = int(
            (self.hparams.data.polar_max_z - self.hparams.data.polar_min_z) / voxel_res[2]) + 1
        self.register_buffer("from_slicegrid_to_sliceindices", torch.tensor([slice_max_r*slice_max_theta*slice_max_z, slice_max_theta*slice_max_z, slice_max_z, 1]))
        

    def __init_pointencoder__(self):
        self.point_encoder = PE.LinearEncoder(
            dim_in=self.hparams.data.input_dim,
            encoder=self.hparams.point_encoder
        )

    @torch.profiler.record_function(f"FORWARD")
    def forward(self, batch, batch_size, batch_idx, kvp_in=None):
        with torch.profiler.record_function(f"VOXELIZE"):
            voxel_ind, point2voxel = compute_point2voxel_map(batch.batch, batch.voxelind)

        with torch.profiler.record_function(f"POINT ENCODER"):
            point_features = self.point_encoder(batch.features)
            
        with torch.profiler.record_function(f"VOXELIZE"):
            voxel_features, voxel_pos = voxelize(
                point_features, batch.pos, point2voxel)

        with torch.profiler.record_function(f"VOXEL ENCODER"):
            voxel_features, kvp_out, maps = self.foward_voxel_encoder(
                voxel_features, voxel_pos, voxel_ind,
                batch_size, batch_idx, point2voxel, batch.pos, kvp_in)

        with torch.profiler.record_function(f"UPSAMPLE"):
            if not hasattr(batch, "backprop"):
                point_features = upsample(point2voxel, point_features, voxel_features)                    
            else:
                point_features = upsample_backprop(batch.backprop, point2voxel, point_features, voxel_features)

        with torch.profiler.record_function(f"POINT_DECODER"):
            point_prediction = self.point_decoder(point_features)
        return point_prediction, point2voxel, maps, kvp_out

    def foward_voxel_encoder(self, voxel_features, voxel_pos, voxel_ind, batch_size, batch_idx, point2voxel, point_pos, kvp_in):

        voxel_features = self.point2voxel_linear(voxel_features)
        spconvgrid = from_grid_to_spconvgrid(voxel_ind, batch_size)

        spconvtensor = spconv.SparseConvTensor(
            voxel_features,
            spconvgrid,
            self.voxel_shape,
            batch_size * self.avail_slices
            )

        voxel_features_skip = []

        for i in range(self.hparams.data.n_hierarchy):
            with torch.profiler.record_function(f"H{i}"):
                spconvtensor = getattr(self, f"H{i}_before")(spconvtensor)
                voxel_features_skip.append(spconvtensor.features)

            with torch.profiler.record_function(f"H{i}_H{i+1}"):
                spconvtensor = getattr(self, f"H{i}_H{i+1}")(spconvtensor)
                spconvtensor = spconvtensor.replace_feature(getattr(self, f"H{i}_H{i+1}_norm")(spconvtensor.features))

        with torch.profiler.record_function(f"POS"):
            voxel_pos, down_spconvindices, slice_indices = retrieve_pos_and_maps(
                voxel_pos, batch_size, spconvgrid, spconvtensor.indices,
                self.from_slicegrid_to_sliceindices, self.full_split_hierarchy)
                  
        with torch.profiler.record_function(f"XYZT"):
            if kvp_in is None:
                kvp_in = {
                    "pos": self.voxel_transformer.zero_kvp_in_pos,
                    "grid": self.voxel_transformer.zero_kvp_in_grid,
                }
                for iblock in range(self.hparams.transformer.n_layers):
                    kvp_in[f"keys_{iblock}"] = self.voxel_transformer.zero_keys
                    kvp_in[f"values_{iblock}"] = self.voxel_transformer.zero_values
                    #setattr(kvp_in, f"keys_{iblock}", self.voxel_transformer.zero_keys)
                    #setattr(kvp_in, f"values_{iblock}", self.voxel_transformer.zero_values)

            newspconvtensorfeatures, probas, iq_innermost, ik_innermost, _, kvp_out = self.voxel_transformer(
                x=spconvtensor.features, xyz=voxel_pos, grid=slice_indices, batch_size=batch_size,
                #in_pos=kvp_in.pos, in_grid=kvp_in.grid, in_keys=kvp_in.keys, in_values=kvp_in.pos)
                kvp_in_isnone=kvp_in is None, kvp_in=kvp_in)
            spconvtensor = spconvtensor.replace_feature(newspconvtensorfeatures)
            self.log_attention(batch_idx, probas, iq_innermost, ik_innermost, voxel_pos, slice_indices, point_pos, point2voxel, maps=[down_spconvindices])

        for i in reversed(range(self.hparams.data.n_hierarchy)):
            with torch.profiler.record_function(f"H{i+1}_H{i}"):
                spconvtensor = getattr(self, f"H{i+1}_H{i}")(spconvtensor)
                spconvtensor = spconvtensor.replace_feature(spconvtensor.features + voxel_features_skip[i])

            with torch.profiler.record_function(f"H{i}"):
                spconvtensor = getattr(self, f"H{i}_after")(spconvtensor)

        return spconvtensor.features, kvp_out, [down_spconvindices]

    def on_train_start(self):
        results = {
            "Loss_total/val": float("nan"), "Loss/point_crossentropy/val": float("nan"), "IoU/val": float("nan"),
            "Loss_total/train": float("nan"), "Loss/point_crossentropy/train": float("nan"), "IoU/train": float("nan")
        }
        if self.hparams.data.eval_in_online_setup:
            results[f'Online_time (ms.)/get_slice/val'] = float("nan")
            results[f'Online_time (ms.)/global_step/val'] = float("nan")
            results[f'Online_time (ms.)/inference/val'] = float("nan")
            results[f'Online_time (ms.)/acquisition/val'] = float("nan")
            results[f'Online_time (ms.)/latency/val'] = float("nan")
            results[f'Online_time (ms.)/can_idle/val'] = float("nan")

        self.logger.log_hyperparams(self.hparams, results)

        self.logger.experiment.add_text(
            "model", self.__repr__().replace("\n", "  \n"), global_step=0)
        
    def global_step(self, batch, batch_idx, tag, kvp_in=None):
        batch_size = batch.seqid.size(0)

        if tag == "train":
            inference_time = time.time()
        else:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)     
            start.record()
            

        if self.hparams.data.temporal_transformer:
            point_oh_pred, point2voxel, maps, kvp_out = self.forward(
                batch, batch_size, batch_idx, kvp_in)
        else:
            point_oh_pred, point2voxel, maps, kvp_out = self.forward(
                batch, batch_size, batch_idx)

        if tag == "train":
            inference_time = time.time() - inference_time
        else:
            end.record()
            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end) / 1000.
            

        with torch.profiler.record_function(f"XE"):
            point_crossentropy = self.criterion_crossentropy(
                point_oh_pred, batch.point_y
            )
        with torch.profiler.record_function(f"LOVASZ"):
            point_lovasz = self.criterion_lovasz(point_oh_pred, batch.point_y)

        out = {
            "loss": point_crossentropy + point_lovasz,
            "point_crossentropy": point_crossentropy.detach(),
            "point_lovasz": point_lovasz.detach(),
            "point_pred": point_oh_pred.detach().argmax(1),
            "inference_time": inference_time,
            "kvp_out": kvp_out,
            "point2voxel": point2voxel.detach(),
            "assignments_maps": maps,
            "batch_size": batch_size
        }
        return out

    def training_step(self, batch, batch_idx):
        return self.global_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        method = "online_step" if self.hparams.data.eval_in_online_setup else "global_step"
        return getattr(self, method)(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        method = "online_step" if self.hparams.data.eval_in_online_setup else "global_step"
        out = getattr(self, method)(batch, batch_idx, "test")

        if self.hparams.data.save_test_preds:
            for b in range(batch.batch.max()+1):
                pred = apply_learning_map(out["point_pred"][batch.batch[batch.backprop]==b], self.hparams.data.learning_map_inv).cpu().numpy().astype(np.uint32)
                pred.tofile(f"sequences/{batch.seqid[b]}/predictions/{batch.scanid[b]:06}.label")     
        return out

    def on_test_epoch_start(self, *args, **kwargs):
        super().on_test_epoch_start(*args, **kwargs)
        self.on_online_epoch_start()    

    def on_test_epoch_end(self, *args, **kwargs):
        super().on_test_epoch_end(*args, **kwargs)
        self.on_online_epoch_end()

    def on_validation_epoch_start(self, *args, **kwargs):
        super().on_validation_epoch_start(*args, **kwargs)
        self.on_online_epoch_start()

    def on_validation_epoch_end(self, *args, **kwargs):
        super().on_validation_epoch_end(*args, **kwargs)
        self.on_online_epoch_end()

    def configure_optimizers(self):

        nowd_lr10_params = [p[1] for p in self.named_parameters() if (
            "_nowd" in p[0]) and ("_lr10" in p[0])]
        nowd_params = [p[1] for p in self.named_parameters() if (
            "_nowd" in p[0]) and ("_lr10" not in p[0])]
        lr10_params = [p[1] for p in self.named_parameters() if (
            "_nowd" not in p[0]) and ("_lr10" in p[0])]
        base_params = [p[1] for p in self.named_parameters() if (
            "_nowd" not in p[0]) and ("_lr10" not in p[0])]

        optimizer = getattr(torch.optim, self.hparams.optim._target_)(
            [
                {'name': f"lr{10*self.hparams.optim.lr}_wd{0}", 'params': nowd_lr10_params,
                    'weight_decay': 0, 'lr': 10 * self.hparams.optim.lr},
                {'name': f"lr{self.hparams.optim.lr}_wd{0}",
                    'params': nowd_params, 'weight_decay': 0},
                {'name': f"lr{10*self.hparams.optim.lr}_wd{self.hparams.optim.weight_decay}",
                    'params': lr10_params, 'lr': 10 * self.hparams.optim.lr},
                {'name': f"lr{self.hparams.optim.lr}_wd{self.hparams.optim.weight_decay}", 'params': base_params}],
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.hparams.lr_patience, min_lr=10e-6),
                "monitor": "IoU/val",
            },
        }