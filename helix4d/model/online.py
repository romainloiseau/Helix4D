import time
import numpy as np
import torch
from tqdm.auto import tqdm

from torch_geometric.data import Data, Batch

class OnlineModel:
    
    @torch.no_grad()
    def online_step(self, batch, batch_idx, tag):
        avail_scan_sequence = [0] + list(self.hparams.data.temporal_scan_sequence)
        from_avail_scan_sequence_to_scan_bucket = torch.zeros((1+np.max(np.abs(self.hparams.data.temporal_scan_sequence)))) if len(self.hparams.data.temporal_scan_sequence) != 0 else torch.zeros(1)
        for iass, ass in enumerate(avail_scan_sequence):
            from_avail_scan_sequence_to_scan_bucket[-ass] = iass
            
        seqid, scanid = batch.seqid.item(), batch.scanid.item()

        if self.current_seqid is None:
            self.current_seqid = seqid
        elif self.current_seqid != seqid:
            tqdm.write(f"CHANGING FROM SEQ {self.current_seqid} to {seqid}")
            self.on_online_epoch_end()
            self.on_online_epoch_start()
            self.current_seqid = seqid

        if hasattr(batch, "pose"):
            pose = batch.pose[0]
            try:
                invpose = torch.linalg.inv(pose.to(batch.pos.device))
            except:
                invpose = None
            del batch.pose
        else:
            pose = None
            invpose = None
        del batch.ptr
        
        loss, point_crossentropy, point_lovasz = [], [], []
        point_pred = torch.zeros((batch.backprop.sum().item(), ), dtype=torch.uint8, device=self.device)


        #tqdm.write(f"{torch.unique(batch.voxelind[:, 0])}")

        for slice in reversed(range(self.hparams.data.slices_per_rotation)):
            #tqdm.write(f"slice {scanid}, {slice}")

            floatscanid = scanid + (self.hparams.data.slices_per_rotation - slice) / self.hparams.data.slices_per_rotation
            
            get_slice_time, is_slice_y, slice_idx, data_slice = self.get_slice(batch, batch_idx, slice)
            if data_slice.features.size(0) != 0:
                kvp_in = self.get_past_kvp(avail_scan_sequence, from_avail_scan_sequence_to_scan_bucket, floatscanid, invpose)

                acq_time = self.compute_acquisition_time(data_slice, tag)

                global_time = time.time()
                with torch.profiler.record_function(f"GLOBAL STEP"):
                    slice_out = self.global_step(
                        data_slice, slice_idx, tag,
                        kvp_in=kvp_in
                        )
                global_time = time.time() - global_time

                if batch_idx > 25:
                    self.log_times(tag, get_slice_time, acq_time, global_time, slice_out["inference_time"])

                self.save_past_kvp(floatscanid, slice_out["kvp_out"], pose)

                point_pred[is_slice_y] = slice_out["point_pred"].to(torch.uint8).to(point_pred.device)
                loss.append(slice_out["loss"].mean())
                point_crossentropy.append(slice_out["point_crossentropy"].mean())
                point_lovasz.append(slice_out["point_lovasz"].mean())

        if tag=="test" and batch_idx%100 == 0:
            tqdm.write(f"{tag} mIoU\t{100*getattr(self, f'IoU_{tag}').compute().detach().cpu().numpy().mean():.2f}")

        self.clean_kvp(avail_scan_sequence)

        out = {
            "loss": torch.tensor(loss).mean(),
            "point_crossentropy": torch.tensor(point_crossentropy).mean(),
            "point_lovasz": torch.tensor(point_lovasz).mean(),
            "point_pred": point_pred.detach(),
            "batch_size": 1
        }
        return out

    def get_slice(self, batch, batch_idx, slice):
        get_slice_time = time.time()
        with torch.profiler.record_function(f"GET slice"):
            is_slice = batch.voxelind[:, 0]==slice
            is_slice_y = is_slice[batch.backprop]
            slice_idx = batch_idx * self.hparams.data.slices_per_rotation + (self.hparams.data.slices_per_rotation - slice - 1)
            data_slice = {}
            for key in batch.keys:
                if key in ["seqid", "scanid"]:
                    data_slice[key] = getattr(batch, key)
                elif key in ["point_y", "point_inst"]:
                    data_slice[key] = getattr(batch, key)[is_slice_y]
                elif key == "time":
                    if getattr(batch, key).size(0) == batch["features"].size(0):
                        data_slice[key] = getattr(batch, key)[is_slice]
                    else:
                        data_slice[key] = getattr(batch, key)
                else:
                    data_slice[key] = getattr(batch, key)[is_slice]

            data_slice["voxelind"][:, 0] = 0
            data_slice = Data(**data_slice)
        get_slice_time = time.time() - get_slice_time
        return get_slice_time,is_slice_y,slice_idx,data_slice

    def log_times(self, tag, get_slice_time, acq_time, global_time, inference_time):
        self.log(f'Online_time (ms.)/get_slice/{tag}',
                1000*get_slice_time, on_step=False, on_epoch=True, batch_size=1)
        self.log(f'Online_time (ms.)/global_step/{tag}',
                1000*global_time, on_step=False, on_epoch=True, batch_size=1)
        self.log(f'Online_time (ms.)/inference/{tag}',
                1000*inference_time, on_step=False, on_epoch=True, batch_size=1)
        self.log(f'Online_time (ms.)/acquisition/{tag}',
                1000*acq_time, on_step=False, on_epoch=True, batch_size=1)
        self.log(f'Online_time (ms.)/latency/{tag}',
                1000*(inference_time + acq_time), on_step=False, on_epoch=True, batch_size=1)
        self.log(f'Online_time (ms.)/can_idle/{tag}',
                1000*(acq_time - inference_time), on_step=False, on_epoch=True, batch_size=1)

    def compute_acquisition_time(self, data_slice, tag):
        if hasattr(data_slice, "time") and data_slice.time.size(0) != 1:
            return data_slice.time.max() - data_slice.time.min()
        return getattr(self.trainer.datamodule, f"{tag}_dataset").sequences[data_slice.seqid[0].item()].dtimes[data_slice.scanid[0].item()] / self.hparams.data.slices_per_rotation

    def from_pos_scan_to_pos(self, pos, pose):
        return torch.matmul(torch.cat((pos, torch.ones((pos.shape[0], 1), dtype=pos.dtype, device=pos.device)), -1), pose[:-1].T)

    def save_past_kvp(self, scanid, kvp_out, pose):
        #tqdm.write(f"out scan\t{scanid:.2f}")
        with torch.profiler.record_function(f"PAST K/V/P OUT"):
            if kvp_out is not None:
                kvp_out = Data(**kvp_out)
                kvp_out.pos = torch.cat([
                        scanid*torch.ones((kvp_out.pos.size(0), 1), dtype=kvp_out.pos.dtype, device=kvp_out.pos.device),
                        self.from_pos_scan_to_pos(kvp_out.pos, pose) if pose is not None else kvp_out.pos
                        ], -1)
                self.kvp_kept_scanid = np.append(self.kvp_kept_scanid, scanid)
                self.kvp_kept.append(kvp_out)

    def get_past_kvp(self, avail_scan_sequence, from_avail_scan_sequence_to_scan_bucket, scanid, invpose):
        with torch.profiler.record_function(f"PAST K/V/P IN"):
            if self.hparams.data.temporal_transformer:
                kvp_in = None
                if len(self.kvp_kept) != 0:
                    keep_slices = np.isin((self.kvp_kept_scanid - scanid - 0.001).astype(int), avail_scan_sequence)
                    #tqdm.write(f"in  scan\t{scanid:.2f}\t{self.kvp_kept_scanid[keep_slices]}")
                    if keep_slices.sum() != 0:
                        kvp_in = Batch.from_data_list([ti for ti, kb in zip(self.kvp_kept, keep_slices) if kb])
                if kvp_in is not None:
                    if invpose is not None:
                        kvp_in.pos[:, 1:] = self.from_pos_scan_to_pos(kvp_in.pos[:, 1:], invpose)
                    kvp_in.pos[:, 0] -= scanid
                    kvp_in.pos[:, 0] = from_avail_scan_sequence_to_scan_bucket[(-kvp_in.pos[:, 0].int()).long()]
                return kvp_in.to_dict() if kvp_in is not None else kvp_in

    def clean_kvp(self, avail_scan_sequence):
        if len(self.kvp_kept) != 0:
            keep_slices = self.kvp_kept_scanid - self.kvp_kept_scanid.max() + 1 > np.min(avail_scan_sequence)
            self.kvp_kept = [ti for ti, kb in zip(self.kvp_kept, keep_slices) if kb]
            self.kvp_kept_scanid = self.kvp_kept_scanid[keep_slices]
    
    def on_online_epoch_start(self, *args, **kwargs):
        if self.hparams.data.eval_in_online_setup:
            self.kvp_kept = []
            self.kvp_kept_scanid = np.array([], dtype=int)
            self.current_seqid = None

    def on_online_epoch_end(self, *args, **kwargs):
        if self.hparams.data.eval_in_online_setup:
            del self.kvp_kept, self.kvp_kept_scanid, self.current_seqid