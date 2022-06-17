import torch
import torch.nn.functional as F

class LovaszLoss(torch.jit.ScriptModule):

    def __init__(self, n_classes=20, ignore_index=0):
        super().__init__()

        self.register_buffer("ignore_index", torch.tensor(ignore_index))
        self.register_buffer("class_to_sum", torch.arange(n_classes, dtype=torch.int64))
        
    @torch.jit.script_method
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        probas = F.softmax(logits, dim=-1)
        
        valid = labels != self.ignore_index

        vprobas = probas[valid]
        vlabels = labels[valid]

        if vprobas.numel() == 0:
            return probas * 0.

        all_fg = (vlabels.unsqueeze(-1) == self.class_to_sum).float()
        all_gts = all_fg.sum(0)
        has_label = all_gts != 0

        all_fg = all_fg[:, has_label]
        vprobas = vprobas[:, has_label]
        all_gts = all_gts[has_label]       

        all_errors = (all_fg - vprobas).abs()

        all_errors_sorted, all_perm = torch.sort(all_errors, 0, descending=True)
        all_perm = all_perm.data

        all_fg_sorted = torch.gather(all_fg, 0, all_perm)
        
        all_intersection = all_gts - all_fg_sorted.cumsum(0)
        all_union = all_gts + (1. - all_fg_sorted).cumsum(0)

        all_jaccard = 1. - all_intersection / all_union
        all_jaccard[1:] = all_jaccard[1:] - all_jaccard[:-1]

        losses = (all_errors_sorted * all_jaccard).sum(0)

        return losses.mean()