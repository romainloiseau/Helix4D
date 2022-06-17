import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_scatter
from matplotlib import cm
from torch import nn

from helix4d.utils import (PROJ_FOV_XY, PROJ_UP_H, do_focus_projection,
                           do_range_projection, do_up_projection)


class LoggingModel:
    #############################################################################
    ### All following methods are ment for monitoring Helix4D in tensorboard ####
    #############################################################################
    def do_greedy_step(self):
        return (self.current_epoch % int(self.hparams.log_every_n_epochs)) == 0

    def on_train_start(self):
        if self.hparams.transformer.do_relative_positional:
            with torch.no_grad():
                avail_dr = torch.linspace(-self.hparams.transformer.spatial_receptive_field-2, self.hparams.transformer.spatial_receptive_field+2, 1000).unsqueeze(-1)
                
                for ixyz, xyz in enumerate(["x", "y", "z"]):
                    
                    avail_dr3 = torch.cat([
                        avail_dr-10000*float(ixyz!=0),
                        avail_dr-10000*float(ixyz!=1),
                        avail_dr-10000*float(ixyz!=2)
                        ], -1).to(self.voxel_transformer.mul_buckets.device)

                    buckets = self.voxel_transformer.from_dxyz_to_bucket(avail_dr3, self.voxel_transformer.mul_buckets,
                        self.voxel_transformer.alpha,
                        self.voxel_transformer.beta,
                        self.voxel_transformer.gamma,
                        self.voxel_transformer.gresolution
                        ).detach().cpu().numpy()

                    figure = plt.figure()
                    plt.plot(avail_dr.detach().cpu().numpy(), buckets)

                    plt.title("Relative xyz buckets")
                    plt.xlabel(f'd{xyz}')
                    plt.ylabel('Bucket id')
                    s, (width, height) = figure.canvas.print_to_buffer()
                    plt.clf()
                    plt.close(figure)
                    del figure
                    
                    self.logger.experiment.add_image(
                        f"Transformer/positional_relative_{xyz}", np.fromstring(s, np.uint8).reshape((height, width, 4)),
                        global_step=0, dataformats='HWC'
                    )
                
        return super().on_train_start()

    def log_attention(self, batch_idx, probas, iq, ik, voxel_pos, voxel_ind, point_pos, point2voxel, maps=None):
        with torch.no_grad():
            if self.training and hasattr(self.logger, "experiment"):
                if batch_idx == 0:
                    if hasattr(self.voxel_transformer, "positional"):
                        self.logger.experiment.add_image(
                            f"XYZT/positional_absolute", self.voxel_transformer.positional.get_display(),
                            global_step=self.current_epoch, dataformats='HWC'
                        )

                        self.log(f'XYZT/positional/scale_nowd', self.voxel_transformer.positional.get_scale().data.mean(),
                                 on_step=False, on_epoch=True)
                        self.logger.experiment.add_histogram(
                            f'XYZT/positional/scale_nowd_hist', self.voxel_transformer.positional.get_scale().data, global_step=self.current_epoch)

                    if self.do_greedy_step():
                        do_projection, h, w = do_up_projection, PROJ_UP_H, PROJ_UP_H
                        batch_size = voxel_ind[:, 0].detach().cpu().max() + 1

                        for n, block in enumerate(self.voxel_transformer.blocks):
                            self.logger.experiment.add_histogram(f'XYZT/temperature/{n}', block.attention.get_T().detach().data, global_step=self.current_epoch)
                            if hasattr(block.attention, "relative_positional_encoding_nowd"):
                                if isinstance(block.attention.relative_positional_encoding_nowd, nn.ParameterList):
                                    for idrpenowd, rpenowd in enumerate(block.attention.relative_positional_encoding_nowd):
                                        self.logger.experiment.add_histogram(f'XYZT/relative_posenc/{n}/{idrpenowd}', rpenowd, global_step=self.current_epoch)
                                else:
                                    self.logger.experiment.add_histogram(f'XYZT/relative_posenc/{n}', block.attention.relative_positional_encoding_nowd, global_step=self.current_epoch)

                        for n, proba in enumerate(probas):

                            if self.hparams.transformer.DEBUG:
                                if hasattr(self.voxel_transformer.blocks[n], "feed_forward"):
                                    for ilayer, layer in enumerate(self.voxel_transformer.blocks[n].feed_forward):
                                        if isinstance(layer, torch.nn.Linear):
                                            self.logger.experiment.add_histogram(
                                                f"XYZT_block{n}_param/FF_weight{ilayer}",
                                                layer.weight,
                                                global_step=self.current_epoch
                                            )
                                            self.logger.experiment.add_histogram(
                                                f"XYZT_block{n}_param/FF_bias{ilayer}",
                                                layer.bias,
                                                global_step=self.current_epoch
                                            )

                                for qkvo in ["q", "k", "v", "o"]:
                                    if hasattr(self.voxel_transformer.blocks[n].attention, f"{qkvo}_linear"):
                                        self.logger.experiment.add_histogram(
                                            f"XYZT_block{n}_param/{qkvo}",
                                            getattr(self.voxel_transformer.blocks[n].attention, f"{qkvo}_linear").weight,
                                            global_step=self.current_epoch
                                        )

                            self.logger.experiment.add_histogram(
                                f"XYZT_proba/{n}", proba, global_step=self.current_epoch)
                            self.logger.experiment.add_histogram(
                                f"XYZT_proba/{n}_max_received", torch_scatter.scatter_max(proba, iq, 0)[0], global_step=self.current_epoch)
                            

                            batch_size_printed = min(batch_size, self.hparams.transformer.N_LOG_ITEM)

                        allproba = torch.stack(probas[:self.hparams.transformer.N_LOG_ATT], -1)

                        batch_point2voxel, equals, frame_ids, projections, highposs, not_projecteds = [], [], [], [], [], []

                        voxel_frame_id = torch.div(voxel_ind[:, 1], self.hparams.data.slices_per_rotation, rounding_mode='floor')
                        n_frameid = voxel_frame_id.max() + 1

                        if maps is not None:
                            if len(maps) == 1:
                                point2voxel = torch.unique(maps[0], return_inverse=True)[-1][point2voxel]
                            else:
                                for map in maps:
                                    point2voxel = map[point2voxel]

                        point_frame_id = voxel_frame_id[point2voxel]
                        voxel_batch = voxel_ind[:, 0].detach().cpu().long()
                        point_batch = voxel_batch[point2voxel].long()

                        point_pos = point_pos.detach().cpu()
                        voxel_pos = voxel_pos.detach().cpu()
                        mean_pos = torch_scatter.scatter_mean(voxel_pos[voxel_frame_id==0], voxel_batch[voxel_frame_id==0], 0)
                        point_pos = point_pos - mean_pos[point_batch]
                        voxel_pos = voxel_pos - mean_pos[voxel_batch]

                        for b in range(batch_size_printed):
                            equals.append(point_batch == b)
                            frame_ids.append(point_frame_id[equals[-1]])

                            highposs.append([point_pos[equals[-1]][frame_ids[-1] == frame_id] for frame_id in range(n_frameid)])
                            projections.append([do_projection(p) for p in highposs[-1]])
                            not_projecteds.append([p < 0 for p in projections[-1]])

                            batch_point2voxel.append([point2voxel[equals[-1]][frame_ids[-1] == frame_id] for frame_id in range(n_frameid)])
                        
                        printvoxel_cond = np.logical_and(voxel_frame_id.detach().cpu().numpy()==0, voxel_pos[:, :2].abs().max(-1)[0].detach().cpu().numpy() < PROJ_FOV_XY)
                        P = []
                        for b in range(batch_size_printed):
                            p = np.logical_and(printvoxel_cond, (voxel_batch==b).numpy()).astype(float)
                            p = p/p.sum()
                            P.append(p/p.sum())

                        printvoxel = np.concatenate([
                            np.random.choice(
                                voxel_ind.size(0),
                                self.hparams.transformer.VIDEO_SUPERVOXEL,
                                replace=False,
                                p=p/p.sum()
                            ) for p in P])
                        
                        for DO_FOCUS_PROJECTION in self.hparams.transformer.do_focus_projection:
                            videos = [None for _ in range(batch_size_printed)]
                            for query in range(voxel_ind.size(0)):
                                batch_query = voxel_batch[query]
                                if batch_query < batch_size_printed:
                                    query_frame_id = voxel_frame_id[query]

                                    if query in printvoxel:
                                        iskey = torch.logical_and(iq == query, ik < voxel_ind.size(0))

                                        key = ik[iskey]

                                        p = allproba[iskey]

                                        colored_images = None

                                        for frame_id in reversed(range(n_frameid)):

                                            image = torch.zeros(
                                                (voxel_batch.size(0), p.size(-2), p.size(-1))).type_as(p)
                                            image[key] = p

                                            image = image[batch_point2voxel[batch_query][frame_id]]

                                            if len(image) != 0:

                                                p_is_0 = (
                                                    image != 0).unsqueeze(-1).detach().cpu()
                                                image = image/image.max(0)[0]

                                                colored_image = torch.from_numpy(cm.get_cmap(
                                                    "coolwarm")(image.detach().cpu().numpy())[..., :-1])

                                                colored_image = colored_image * p_is_0

                                                #seg_image = self.from_labels_to_color(highsegs[batch_query][frame_id], dark=True).unsqueeze(1).unsqueeze(1)
                                                #colored_image = colored_image + ~p_is_0*seg_image

                                                if DO_FOCUS_PROJECTION:
                                                    # highpos[equals[batch_query]][frame_ids[batch_query]==frame_id]
                                                    position_to_project = highposs[batch_query][frame_id]
                                                    focus_id = (
                                                        (position_to_project - voxel_pos[query])**2).sum(-1).argmin()
                                                    projection = do_focus_projection(
                                                        position_to_project, focus_id=focus_id)
                                                    colored_image = colored_image[projection]
                                                    colored_image[projection < 0] = 1.
                                                else:
                                                    colored_image = colored_image[projections[batch_query][frame_id]]
                                                    colored_image[not_projecteds[batch_query]
                                                                [frame_id]] = 1.

                                            else:
                                                colored_image = torch.ones(
                                                    (h, w, p.size(-2), p.size(-1), 3))

                                            if query_frame_id == frame_id:
                                                if DO_FOCUS_PROJECTION:
                                                    inside = torch.zeros(
                                                        (h, w)).bool()
                                                    inside[h//2, w//2] = True
                                                else:
                                                    inside = torch.from_numpy(do_projection(
                                                        voxel_pos[query].unsqueeze(0)) != -1)
                                                for _ in range(2):
                                                    inside = self.dilatation_image(
                                                        inside, torch.logical_or)
                                                contours = self.get_contours(
                                                    colored_image, inside)
                                                contours = self.dilatation_image(
                                                    contours, torch.maximum)

                                                contours = contours.unsqueeze(
                                                    -1).unsqueeze(-1).unsqueeze(-1)
                                                colored_image = (1-contours.float()) * colored_image + contours.float(
                                                ) * torch.tensor([0., 1., 0.], device=colored_image.device).view(1, 1, 1, 1, -1)

                                            colored_image = (
                                                255*colored_image).to(torch.uint8)

                                            if colored_images is None:
                                                colored_images = colored_image.unsqueeze(
                                                    0)
                                            else:
                                                colored_images = torch.cat(
                                                    [colored_images, colored_image.unsqueeze(0)], 0)

                                        if videos[batch_query] is None:
                                            videos[batch_query] = colored_images.unsqueeze(
                                                0)
                                        else:
                                            videos[batch_query] = torch.cat(
                                                [videos[batch_query], colored_images.unsqueeze(0)], 0)


                            for b in range(batch_size_printed):
                                video = videos[b].permute(
                                    5, 0, 6, 4, 2, 1, 3)                                    
                                    
                                video = video.flatten(-2, -1).flatten(-3, -2).unsqueeze(1)

                                nokeep = video[0].flatten(0, -2).min(0)[0] == 255
                                start, end = 0, nokeep.size(-1)
                                while nokeep[start] and start < end:
                                    start += 1
                                while nokeep[end-1] and start < end:
                                    end -= 1
                                video = video[..., start:end]

                                for n, videon in enumerate(video):
                                    label = "FOCUS" if DO_FOCUS_PROJECTION else "BEV"
                                    self.logger.experiment.add_video(
                                        f"attention_{label}_{n}/{b}", videon, global_step=self.current_epoch, fps=1 / (1. + DO_FOCUS_PROJECTION))

    def get_contours(self, colored_image, inside):
        contours = torch.cat([inside[1:], 0*inside[:1]], 0) + torch.cat([inside[:-1], 0*inside[-1:]], 0) + torch.cat(
            [inside[:, 1:], 0*inside[:, :1]], 1) + torch.cat([inside[:, :-1], 0*inside[:, -1:]], 1)
        contours[1:, 1:] = contours[1:, 1:] + inside[:-1, :-1]
        contours[:-1, :-1] = contours[:-1, :-1] + inside[1:, 1:]
        contours[:-1, 1:] = contours[:-1, 1:] + inside[1:, :-1]
        contours[1:, :-1] = contours[1:, :-1] + inside[:-1, 1:]
        contours = (contours * (1-inside.float()) > 0).to(colored_image.device)
        return contours

    def dilatation_image(self, inside, func=torch.logical_or):
        inside[:-2, :-2] = func(inside[1:-1, 1:-1], inside[:-2, :-2])
        inside[1:-1, :-2] = func(inside[1:-1, 1:-1], inside[1:-1, :-2])
        inside[2:, :-2] = func(inside[1:-1, 1:-1], inside[2:, :-2])
        inside[:-2, 2:] = func(inside[1:-1, 1:-1], inside[:-2, 2:])
        inside[1:-1, 2:] = func(inside[1:-1, 1:-1], inside[1:-1, 2:])
        inside[2:, 2:] = func(inside[1:-1, 1:-1], inside[2:, 2:])
        inside[:-2, 1:-1] = func(inside[1:-1, 1:-1], inside[:-2, 1:-1])
        inside[2:, 1:-1] = func(inside[1:-1, 1:-1], inside[2:, 1:-1])

        return inside
