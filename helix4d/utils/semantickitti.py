import copy

import torch

import numpy as np

def apply_learning_map(segmentation, f):
    mapped = copy.deepcopy(segmentation)
    for k, v in f.items():
        mapped[segmentation == k] = v
    return mapped

def from_sem_to_color(segmentation, f):
    color = torch.zeros(list(segmentation.size())+[3]).int()
    color[..., 0] = 255
    for k, v in f.items():
        color[segmentation == k] = torch.tensor(v).int()
    return color

def from_inst_to_color(instance):
    max_inst_id = 100000
    inst_color = np.random.uniform(low=0.0,
                                   high=1.0,
                                   size=(max_inst_id, 3))
    inst_color[0] = np.full((3), 0.1)
    inst_color = torch.tensor(255*inst_color).int()
    
    color = torch.zeros(list(instance.size())+[3]).int()
    for k in torch.unique(instance.flatten()):
        color[instance == k] = inst_color[k]
    return color