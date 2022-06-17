import itertools
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from helix4d.utils import (PROJ_FOV_XY, PROJ_UP_H, apply_learning_map,
                           do_focus_projection, do_range_projection,
                           do_up_projection, from_sem_to_color)
from matplotlib import cm
from pytorch_lightning.callbacks import Callback


class BaseLogger(Callback):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.hparams = SimpleNamespace(**kwargs)

        self.class_names = [self.hparams.labels[int(label)] for label in apply_learning_map(
            np.arange(self.hparams.output_dim), self.hparams.learning_map_inv)][1:]

    def do_greedy_step(self, epoch):
        return (epoch % int(self.hparams.log_every_n_epochs)) == 0

    def from_labels_to_color(self, labels):
        return from_sem_to_color(apply_learning_map(labels, self.hparams.learning_map_inv), self.hparams.color_map)

    def greedy_images(self, moduletensorboard, current_epoch, batch, tag, prediction, *args, **kwargs):

        for do_projection, view, h, w in zip([do_up_projection], ["top_view"], [PROJ_UP_H], [PROJ_UP_H]):
            images = []

            do_assignments = ("assignments" in kwargs.keys()) and (
                kwargs["assignments"] is not None)
            assignments = []

            backprop = batch.backprop.detach().cpu()
            prediction = prediction.detach().cpu()

            pos = batch.pos.detach().cpu()[backprop]
            #pos = pos-pos.mean(0) * torch.tensor([1, 1, 0])
            #point_y = batch.point_y.detach().cpu()[backprop]
            point_y = batch.point_y.detach().cpu()
            ibatch = batch.batch.detach().cpu()[backprop]
            
            for i in range(ibatch.max() + 1):
                equal = ibatch == i
                pos[equal] = pos[equal]-pos[equal].mean(0)
                projection = do_projection(pos[equal])

                not_projected = projection < 0

                true = point_y[equal][projection]
                pred = prediction[equal][projection]
                c = 300 + (70 - 300) * (true == pred)
                c[true == 0] = 0
                
                true = self.from_labels_to_color(true) / 255.
                pred = self.from_labels_to_color(pred.long()) / 255.
                c = self.from_labels_to_color(c) / 255.
                true[not_projected] = 1
                pred[not_projected] = 1
                c[not_projected] = 1
                images.append(torch.cat([true, pred, c], 0))

                if do_assignments:
                    point2voxel = kwargs["assignments"]
                    if ("assignments_maps" in kwargs.keys()) and (kwargs["assignments_maps"] is not None):
                        for map in kwargs["assignments_maps"]:
                            point2voxel = map[point2voxel]
                    superpoints = point2voxel[backprop][equal][projection]
                    #superpoints = superpoints - i*np.prod(self.hparams.voxel_shape)
                    superpoints = superpoints - superpoints.min()
                    superpoints = torch.randperm(superpoints.max()+1)[
                        superpoints] % 20
                    superpoints = torch.from_numpy(cm.get_cmap("tab20")(
                        superpoints.detach().cpu().numpy())[..., :-1])
                    superpoints[not_projected] = 1
                    assignments.append(superpoints)

            images = torch.stack(images)

            if do_assignments:
                assignments = torch.stack(assignments)
                images = torch.cat([images, assignments], 1)

            if view == "sphere":
                images = images.view(
                    1, -1, (3 + do_assignments) * h, w, 3).flatten(1, 2)
            else:
                images = images.view(-1, 1, (3 + do_assignments)
                                     * h, w, 3).flatten(1, 2)
            images = images.permute(1, 0, 2, 3).flatten(1, 2)

            moduletensorboard.add_image(
                f"{tag}/{view}_{batch.seqid.cpu().numpy()}_{batch.scanid.cpu().numpy()}", images,
                global_step=current_epoch, dataformats='HWC'
            )

    def image_confusion_matrix(self, cm, curr_iou):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """

        cm = cm[1:, 1:]

        curr_iou = curr_iou[:, np.newaxis]
        n_samples = cm.sum(axis=1)[:, np.newaxis]

        cm = np.nan_to_num(cm.astype('float') / n_samples)

        n_samples = n_samples/n_samples.sum()

        cm = np.hstack((cm, 0 * curr_iou, curr_iou, n_samples))

        figure = plt.figure(figsize=(
            2 + (self.hparams.output_dim + 3) * 7 / 20.,
            1 + self.hparams.output_dim * 7 / 20.
            ))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        yticks_marks = np.arange(len(self.class_names))
        xticks_marks = np.hstack(
            (yticks_marks, len(self.class_names) + 1, len(self.class_names) + 2))
        plt.tick_params(labelright=True, right=True)
        plt.xticks(xticks_marks, self.class_names + ["IoU", "N%"], rotation=60)
        plt.yticks(yticks_marks, self.class_names)

        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if j != cm.shape[0]:
                color = "white" if cm[i, j] > threshold else "black"
                cmfloat = np.around(
                    100 * cm[i, j], decimals=1 if 100 * cm[i, j] >= 10 else 2) if cm[i, j] != 0 else ""
                plt.text(j, i, cmfloat, horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        s, (width, height) = figure.canvas.print_to_buffer()
        plt.clf()
        plt.close(figure)
        del figure
        return np.fromstring(s, np.uint8).reshape((height, width, 4))
