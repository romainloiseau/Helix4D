import torch
import numpy as np

def cart2cyl(input_xyz):
    rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)    
    phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    return np.stack((rho, phi, input_xyz[..., 2]), axis=-1)

def cyl2cart(input_xyz_polar):
    x = input_xyz_polar[..., 0] * np.cos(input_xyz_polar[..., 1])
    y = input_xyz_polar[..., 0] * np.sin(input_xyz_polar[..., 1])
    return np.stack((x, y, input_xyz_polar[..., 2]), axis=-1)

def cart2theta(input_xyz):
    return np.arctan2(input_xyz[..., 1], input_xyz[..., 0])

def cart2cyl_torch(input_xyz):
    rho = torch.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)    
    phi = torch.atan2(input_xyz[..., 1], input_xyz[..., 0])
    return torch.stack((rho, phi, input_xyz[..., 2]), axis=-1)

def cyl2cart_torch(input_xyz_polar):
    x = input_xyz_polar[..., 0] * torch.cos(input_xyz_polar[..., 1])
    y = input_xyz_polar[..., 0] * torch.sin(input_xyz_polar[..., 1])
    return torch.stack((x, y, input_xyz_polar[..., 2]), axis=-1)