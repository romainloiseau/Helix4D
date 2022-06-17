import torch
import numpy as np

proj_H=64
proj_W=1024
proj_fov_up=3.0
proj_fov_down=-25.0

PROJ_UP_H=256
PROJ_FOV_XY=35.

def do_focus_projection(points, focus_id=None, proj_fov_xy=PROJ_FOV_XY, proj_up_h=PROJ_UP_H):
    if focus_id is not None:
        P = points[focus_id]
        c, s = -P[1] / torch.norm(P[:2]), P[0] / torch.norm(P[:2])
        mat = torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        x = points.matmul(mat.T)

        x[:, 2] = x[:, 2] - x[focus_id][2]

        P = x[focus_id]
        Hz = 3
        norm_camera = (Hz**2+P[1]**2)**.5
        c, s = Hz / norm_camera, P[1] / norm_camera
        mat = torch.tensor([[1, 0, 0], [0, c, s], [0, -s, c]])
        x = x.matmul(mat.T)
        
        zoom = torch.norm(P[:2]) / torch.norm(x[focus_id][:2])
        x = x - x[focus_id]

        #x[x[:, 2] > Hz] *= 0
        x *= 4*zoom**.5
        return do_up_projection(x, proj_fov_xy=proj_fov_xy, proj_up_h=proj_up_h)
    return do_up_projection(points, proj_fov_xy=proj_fov_xy, proj_up_h=proj_up_h)


def do_up_projection(points, proj_fov_xy=PROJ_FOV_XY, proj_up_h=PROJ_UP_H):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # get scan components
    scan_x = (1. + points[:, 0] / proj_fov_xy) / 2.
    scan_y = (1. + points[:, 1] / proj_fov_xy) / 2.

    # scale to image size using angular resolution
    scan_x *= proj_up_h+2                              # in [0.0, W]
    scan_y *= proj_up_h+2                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = torch.clamp(torch.floor(scan_x), 0, proj_up_h-1).int().numpy()
    proj_y = torch.clamp(torch.floor(scan_y), 0, proj_up_h-1).int().numpy()

    # order in decreasing depth
    indices = np.arange(points.shape[0])
    order = np.argsort(points[:, -1].numpy())
    indices = indices[order]
    
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    
    proj_idx = np.full((proj_up_h, proj_up_h),
                       -1,
                       dtype=np.int32)
    proj_idx[proj_y, proj_x] = indices
    return proj_idx

def do_range_projection(points):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = (points**2).sum(-1)**.5
    #torch.linalg.norm(
    #    points,
    #    2,
    #    axis=1
    #)
    
    points_cleaned = points[depth > 10e-8]
    depth = depth[depth > 10e-8]
    
    
    # get scan components
    scan_x = points_cleaned[:, 0]
    scan_y = points_cleaned[:, 1]
    scan_z = points_cleaned[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = torch.clamp(torch.floor(proj_x), 0, proj_W - 1).int().numpy()
    proj_y = torch.clamp(torch.floor(proj_y), 0, proj_H - 1).int().numpy()

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth.numpy())[::-1]
    indices = indices[order]
    
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    
    proj_idx = np.full((proj_H, proj_W),
                       -1,
                       dtype=np.int32)
    proj_idx[proj_y, proj_x] = indices
    return proj_idx