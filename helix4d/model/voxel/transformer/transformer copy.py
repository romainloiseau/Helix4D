import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from matplotlib import cm
from torch import nn
from torch_geometric.data import Data

from ...sparseops import LayerNorm
from .block import TransformerBlockFFN, TransformerBlockNOFFN

torch._C._jit_override_can_fuse_on_gpu(False)

@torch.jit.script
def jit_compute_interactions(
    grid:torch.Tensor, xyz:torch.Tensor, kvp_in_pos: torch.Tensor, kvp_in_grid: torch.Tensor,
    force_keep_neighbours: bool, spatial_receptive_field:float, spatial_receptive_field2: float, slices_per_rotation:int
    ):
    query_size = grid.size(0)

    xyz = torch.cat([xyz, kvp_in_pos[:, 1:]], 0)
    grid = torch.cat([grid, kvp_in_grid], 0)
    have_kvp = query_size != grid.size(0)

    assert grid.max() < 128
    grid = grid.to(torch.int8)

    batch = grid[:, 0]
    with torch.profiler.record_function("IJ batch"):
        i = torch.arange(batch.size(
                0), device=batch.device, dtype=torch.int32) #int16 before ! changed for HelixNet !

        if have_kvp:
            iqik = torch.stack(torch.meshgrid((i[:query_size], i),
            ), -1).flatten(0, 1).long()
        else:
            iqik = torch.stack(torch.meshgrid((i, i),
            ), -1).flatten(0, 1)
            iqik = iqik[(batch.unsqueeze(-1) == batch).flatten()].long()

    if not have_kvp:
        block = grid[:, 1]
        with torch.profiler.record_function("IJ block"):
            blockiqik = block[iqik]
            keep_block = blockiqik[:, 0] <= blockiqik[:, 1]
            keep_blocki = torch.where(keep_block)[0]
            iqik = torch.index_select(iqik, 0, keep_blocki)

    with torch.profiler.record_function("SPATIAL"):
        iqtemp, iktemp = iqik[:, 0], iqik[:, 1]

        if force_keep_neighbours:
            grid_look = grid[:, 1:]
            grid_look_iq, grid_look_ik = torch.index_select(
                grid_look, 0, iqtemp), torch.index_select(grid_look, 0, iktemp)
            forcekeep = (grid_look_iq[:, 1:] - grid_look_ik[:, 1:]).abs().max(-1)[0] <= 1

            if have_kvp:
                forcekeep = torch.logical_and(  # rtz <= 1 and first block
                    forcekeep, iktemp < query_size)
            else:
                forcekeep = torch.logical_and(  # rtz <= 1 and same block
                    forcekeep, grid_look_iq[:, 0] == grid_look_ik[:, 0])
        else:
            forcekeep = torch.zeros((grid.size(0), ), dtype=torch.bool, device=grid.device)

        xyz16 = xyz.to(torch.float16)

        # Keep closest points
        if spatial_receptive_field2 != 0:
            
            xy_coarse = torch.ceil(xyz[:, :2] / spatial_receptive_field).to(torch.int8)
            keep_coarse = (torch.index_select(xy_coarse, 0, iqtemp) - torch.index_select(xy_coarse, 0, iktemp)).abs().max(-1)[0] <= 1
            if force_keep_neighbours:
                keep_coarse = torch.logical_or(forcekeep, keep_coarse)
                forcekeep = forcekeep[keep_coarse]

            iqik = iqik[keep_coarse]
            
            dxyz16 = torch.index_select(xyz16, 0, iqik[:, 0]) - torch.index_select(xyz16, 0, iqik[:, 1])
            keepr = (dxyz16**2).sum(-1) <= spatial_receptive_field2
            if force_keep_neighbours:
                keepr = torch.logical_or(forcekeep, keepr)
                forcekeep = forcekeep[keepr]

            iqik = iqik[keepr]
            dxyz16 = dxyz16[keepr]
        else:
            dxyz16 = torch.index_select(xyz16, 0, iqik[:, 0]) - torch.index_select(xyz16, 0, iqik[:, 1])
            
    iq, ik = iqik[:, 0], iqik[:, 1]

    if have_kvp:
        scanid = torch.hstack((torch.zeros(query_size, device=kvp_in_pos[:, 0].device, dtype=torch.int), kvp_in_pos[:, 0].int()))
        dscanid = scanid[ik]
    else:
        scanid = torch.div(grid[:, 1], slices_per_rotation, rounding_mode="floor")
        dscanid = scanid[ik] - scanid[iq]
    
    return iq, ik, (dxyz16, dscanid)

@torch.jit.script
def from_buckets_to_temporalbuckets_mul(n_avail_temporal_block: int, xyz_buckets, dt, mul_buckets):
    return n_avail_temporal_block * xyz_buckets + dt

@torch.jit.script
def from_buckets_to_temporalbuckets_plus(n_avail_temporal_block: int, xyz_buckets, dt, mul_buckets):
    return torch.cat([xyz_buckets, dt.unsqueeze(-1) + mul_buckets], -1)

@torch.jit.script
def compute_batch_for_layernorm(grid, batch_size: int):
    return (grid[:, 0] + batch_size * grid[:, 1]).long()

@torch.jit.script
def from_dxyz_to_bucket_base(x, alpha, beta, gamma, res):
    if res.size(0) == 2:
        x = torch.stack([torch.sqrt(x[:, 0]**2 + x[:, 1]**2), x[:, -1]], -1)
    x = x / res
    x_abs = x.abs()

    sup = torch.clamp(torch.round(alpha - (beta - alpha) * torch.log(x_abs / alpha) / torch.log(gamma / alpha)).to(x.dtype), max=beta)
    sup = torch.sign(x) * sup

    buckets = beta + torch.where(x_abs <= alpha, torch.round(x), sup)
    return buckets

@torch.jit.script
def from_dxyz_to_bucket_mul(x: torch.Tensor, mul_buckets: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, res: torch.Tensor):
    buckets = from_dxyz_to_bucket_base(x, alpha, beta, gamma, res)

    buckets = (mul_buckets * buckets).sum(-1)

    return buckets.long()

@torch.jit.script
def from_dxyz_to_bucket_plus(x: torch.Tensor, mul_buckets: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, res: torch.Tensor):
    buckets = from_dxyz_to_bucket_base(x, alpha, beta, gamma, res).long()

    buckets = buckets + mul_buckets[:buckets.size(-1)]
    return buckets

class Transformer(nn.Module):

    def __init__(self, dim_in, transformer, temporal_transformer=False, slices_per_rotation=1, *args, **kwargs):
        super().__init__()

        self.spatial_receptive_field = transformer.spatial_receptive_field
        self.spatial_receptive_field2 = transformer.spatial_receptive_field**2
        self.force_keep_neighbours = transformer.force_keep_neighbours
        self.temporal_transformer = temporal_transformer
        self.slices_per_rotation = slices_per_rotation
        
        self.register_buffer("zero_kvp_in_pos", torch.zeros((0, 4)))
        self.register_buffer("zero_kvp_in_grid", torch.zeros((0, 5), dtype=torch.int32))

        if transformer.do_relative_positional:
            self.register_buffer("alpha", torch.tensor(
                transformer.relative_positional_alpha, dtype=torch.float16))
            self.register_buffer("beta", torch.tensor(
                transformer.relative_positional_beta, dtype=torch.float16))
            self.register_buffer("gamma", torch.tensor(
                transformer.relative_positional_gamma, dtype=torch.float16))
            self.register_buffer("gresolution", torch.tensor(
                transformer.relative_positional_gresolution, dtype=torch.float16))

            self.relative_positional_mode = transformer.relative_positional_mode
            if transformer.relative_positional_mode == "mul":
                n_buckets = 2*torch.tensor(transformer.relative_positional_beta)+1
                if n_buckets.numel() == 1:
                    mul_buckets = [n_buckets**2, n_buckets, 1]
                elif n_buckets.numel() == 2:
                    mul_buckets = [n_buckets[-1], 1]
                elif n_buckets.numel() == 3:
                    mul_buckets = [n_buckets[-1]*n_buckets[-2], n_buckets[-1], 1]
                else:
                    raise ValueError

                self.register_buffer("mul_buckets", torch.tensor(mul_buckets, dtype=torch.float16))

                self.from_dxyz_to_bucket = from_dxyz_to_bucket_mul

            elif transformer.relative_positional_mode == "plus":
                n_buckets = list(2*np.array(transformer.relative_positional_beta)+1)

                if transformer.relative_positional_ntime != 1:
                    n_buckets += [transformer.relative_positional_ntime]

                self.register_buffer("mul_buckets", torch.tensor([0] + list(np.cumsum(n_buckets))[:-1], dtype=torch.int64))

                self.from_dxyz_to_bucket = from_dxyz_to_bucket_plus

            self.temporal_buckets = transformer.temporal_buckets
            if self.temporal_buckets:
                self.n_avail_temporal_block = transformer.relative_positional_ntime

                if transformer.relative_positional_mode == "mul":
                    self.from_buckets_to_temporalbuckets = from_buckets_to_temporalbuckets_mul
                elif transformer.relative_positional_mode == "plus":
                    self.from_buckets_to_temporalbuckets = from_buckets_to_temporalbuckets_plus
                else:
                    raise ValueError


        TransformerBlock = TransformerBlockFFN if transformer.do_ffn else TransformerBlockNOFFN
        self.blocks = nn.ModuleList([TransformerBlock(
            dim_in, transformer, i) for i in range(transformer.n_layers)])

        self.do_relative_positional = transformer.do_relative_positional

        self.register_buffer("zero_keys", torch.zeros(
            (0, transformer.n_heads, transformer.dim_qk if transformer.dim_qk != 0 else dim_in // transformer.n_heads)))
        self.register_buffer("zero_values", torch.zeros(
            (0, transformer.n_heads, dim_in // transformer.n_heads)))
        self.register_buffer("zero_batch", torch.zeros(0))

        if transformer.batchnorm_output:
            self.batchnorm_output = nn.BatchNorm1d(dim_in)

        if transformer.architecture == "Pre-LN":
            self.norm_output = LayerNorm(dim_in)
        elif transformer.architecture == "Post-LN":
            self.norm_input = LayerNorm(dim_in)
        else:
            raise ValueError(
                f'Invalid architecture "{transformer.architecture}", should be in ["Pre-LN", "Post-LN"]')

    def forward(self, x: torch.Tensor, xyz: torch.Tensor, grid: torch.Tensor, batch_size: int, kvp_in_isnone: bool, iq=None, ik=None, buckets=None, kvp_in=None, logger=None):
        probas = []

        if kvp_in_isnone:
            batch_for_layernorm = compute_batch_for_layernorm(grid, batch_size)
        else:
            batch_for_layernorm = self.zero_batch

        if logger is not None:
            logger["tb"].add_histogram(
                f"{logger['name']}_forward/0_inp", x, global_step=logger["global_step"])

        if iq is None or ik is None:
            with torch.no_grad():
                iq, ik, dxyz16 = self.compute_interactions(
                    grid, xyz, batch_size, kvp_in)

        if self.do_relative_positional and buckets is None:
            buckets = self.compute_buckets(dxyz16)
        else:
            buckets = torch.tensor([]).to(x.device)

        kvp_out = self.prepare_kvpos_out(xyz, grid)

        if hasattr(self, "norm_input"):
            with torch.profiler.record_function("NORM"):
                x = self.norm_input(x, batch=batch_for_layernorm)
                if logger is not None:
                    logger["tb"].add_histogram(
                        f"{logger['name']}_forward/3_inp+pos+norm", x, global_step=logger["global_step"])

        for iblock, block in enumerate(self.blocks):
            if logger is not None:
                logger["iblock"] = iblock
            if kvp_in is not None:
                x, proba, key, value = block(
                    x, batch_for_layernorm, iq, ik, buckets=buckets,
                    keystouse=getattr(
                        kvp_in, f"keys_{iblock}"),
                    valuestouse=getattr(
                        kvp_in, f"values_{iblock}"))
            else:
                x, proba, key, value = block(x, batch_for_layernorm, iq, ik, buckets=buckets, keystouse=self.zero_keys, valuestouse=self.zero_values)

            if logger is not None and self.do_relative_positional:
                with torch.no_grad():
                    shifts = self.get_shifts_for_vizu()
                    if shifts is not None and self.relative_positional_mode=="mul":

                        if hasattr(self, "n_avail_temporal_block"):
                            dtime = buckets % self.n_avail_temporal_block
                            xyz_buckets = torch.div(buckets, self.n_avail_temporal_block, rounding_mode="floor")

                            count = torch_scatter.scatter_add((proba >= -1).float(), buckets, 0, dim_size=self.n_avail_temporal_block*shifts.shape[0])
                            keep_count = (count >= 2.*count.max()/100.).float()
                            mean_count = count / count.max(0)[0]

                            for dt in range(self.n_avail_temporal_block):
                                keep_dt = dtime==dt
                                this_dt_bucket = xyz_buckets[keep_dt]
                                this_dt_proba = proba[keep_dt]
                                if logger['iblock'] == 0:
                                    self.print_kernel_attention_count(mean_count[dt::self.n_avail_temporal_block], logger, shifts, dt=dt)

                                self.print_kernel_attention(this_dt_bucket, logger, this_dt_proba, shifts, keep_count[dt::self.n_avail_temporal_block], dt=dt)
                        else:
                            if logger['iblock'] == 0:
                                count = torch_scatter.scatter_add((proba >= -1).float(), buckets, 0, dim_size=shifts.shape[0])
                                keep_count = (count >= 2.*count.max()/100.).float()
                                self.print_kernel_attention_count(count / count.max(0)[0], logger, shifts)
                            self.print_kernel_attention(buckets, logger, proba, shifts, keep_count)

            probas.append(proba)
            self.update_kvpos_out(kvp_out, iblock, key, value)

        if logger is not None:
            logger["tb"].add_histogram(
                f"{logger['name']}_forward/4_inp+pos+norm+T", x, global_step=logger["global_step"])

        if hasattr(self, "norm_output"):
            x = self.norm_output(x, batch=batch_for_layernorm)
            if logger is not None:
                logger["tb"].add_histogram(
                    f"{logger['name']}_forward/5_inp+pos+norm+T+norm", x, global_step=logger["global_step"])

        if hasattr(self, "batchnorm_output"):
            x = self.batchnorm_output(x)
            if logger is not None:
                logger["tb"].add_histogram(
                    f"{logger['name']}_forward/5_inp+pos+norm+T+batchnorm", x, global_step=logger["global_step"])

        return x, probas, iq, ik, buckets, kvp_out        

    def print_kernel_attention_count(self, mean, logger, shifts, dt=None):
        is_3x3x3 = shifts.size(0) == 27

        image_size = tuple(torch.cat((shifts.max(0)[0].detach().cpu() + 1, torch.tensor([mean.size(-1)]))))
        image = torch.zeros(image_size, device=mean.device, dtype=mean.dtype)

        image[shifts[:, 0], shifts[:, 1], shifts[:, 2]] = mean

        repeat_interleave = 16 if is_3x3x3 else 2
        image = torch.repeat_interleave(image, repeat_interleave, dim=0)
        image = torch.repeat_interleave(image, repeat_interleave, dim=1)
        image = image.detach().cpu()

        image = torch.from_numpy(cm.get_cmap("bwr")(image.detach().cpu().numpy())).permute(4, 2, 3, 0, 1)
        image = F.pad(image, (1, 1, 1, 1), "constant", 0)
        image = image.permute(0, 1, 3, 2, 4).flatten(1, 2).flatten(2, 3)
                            
        logger["tb"].add_image(
                                f"attention_{logger['name']}/sum" + (f"_dt{dt}" if dt is not None else ""), image,
                                    global_step=logger["global_step"], dataformats='CHW'
                            )

    def print_kernel_attention(self, buckets, logger, proba, shifts, keep_count, dt=None):
        mean = torch_scatter.scatter_mean(proba, buckets, 0, dim_size=shifts.shape[0]) * keep_count

        is_3x3x3 = shifts.size(0) == 27

        if not is_3x3x3:
            mean /= mean.max(0)[0]

        image_size = tuple(torch.cat((shifts.max(0)[0].detach().cpu() + 1, torch.tensor([mean.size(-1)]))))
        image = torch.zeros(image_size, device=mean.device, dtype=mean.dtype)

        image[shifts[:, 0], shifts[:, 1], shifts[:, 2]] = mean

        repeat_interleave = 16 if is_3x3x3 else 2
        image = torch.repeat_interleave(image, repeat_interleave, dim=0)
        image = torch.repeat_interleave(image, repeat_interleave, dim=1)
        image = image.detach().cpu()

        image = torch.from_numpy(cm.get_cmap("bwr")(image.detach().cpu().numpy())).permute(4, 2, 3, 0, 1)
        image = F.pad(image, (1, 1, 1, 1), "constant", 0)
        image = image.permute(0, 1, 3, 2, 4).flatten(1, 2).flatten(2, 3)
                        
        logger["tb"].add_image(
                            f"attention_{logger['name']}/{logger['iblock']}" + (f"_dt{dt}" if dt is not None else ""), image,
                                global_step=logger["global_step"], dataformats='CHW'
                        )

    def get_shifts_for_vizu(self):
        if self.beta.numel() == 1:
            shifts_rtz = torch.stack(torch.meshgrid(*([torch.arange(2*self.beta + 1, device=self.mul_buckets.device) for _ in range(3)]), 
            #indexing="ij"
            ), -1).flatten(0, -2).long()
        elif self.beta.numel() == 2:
            shifts_rtz = None
        elif self.beta.numel() == 3:
            shifts_rtz = torch.stack(torch.meshgrid(*([torch.arange(2*beta + 1, device=self.mul_buckets.device) for beta in self.beta]), 
            #indexing="ij"
            ), -1).flatten(0, -2).long()
        
        return shifts_rtz

    def compute_buckets(self, dxyz16):

        xyz_buckets = self.from_dxyz_to_bucket(dxyz16[0], self.mul_buckets, self.alpha, self.beta, self.gamma, self.gresolution)
        #print(xyz_buckets)

        if self.temporal_buckets:
            return self.from_buckets_to_temporalbuckets(self.n_avail_temporal_block, xyz_buckets, dxyz16[1], self.mul_buckets[-1])

        return xyz_buckets

    def prepare_kvpos_out(self, xyz, grid):
        kvp_out = None
        if self.temporal_transformer:
            kvp_out = Data(pos=xyz, grid=grid)
        return kvp_out

    def update_kvpos_out(self, kvp_out, iblock, key, value):
        if self.temporal_transformer:
            setattr(kvp_out, f"keys_{iblock}", key)
            setattr(kvp_out, f"values_{iblock}", value)

    def compute_interactions(self, grid: torch.Tensor, xyz: torch.Tensor, batch_size: int, kvp_in):
        with torch.profiler.record_function("IJ"):
            return jit_compute_interactions(
                grid, xyz,
                kvp_in_pos=kvp_in.pos if kvp_in is not None else self.zero_kvp_in_pos,
                kvp_in_grid=kvp_in.grid if kvp_in is not None else self.zero_kvp_in_grid,
                force_keep_neighbours=self.force_keep_neighbours,
                spatial_receptive_field=self.spatial_receptive_field,
                spatial_receptive_field2=self.spatial_receptive_field2,
                slices_per_rotation=self.slices_per_rotation)
