import os
from glob import glob
import torch
from torch.nn import functional as F
import numpy as np 
import utils.encode as utils_encode 

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def make_grid(grid_size, grid_res, grid_offset):
    """
    Constructs an array representing the corners of a grid
    """

    width, height, depth = grid_size
    x_grid_res, y_grid_res, z_grid_res = grid_res
    xoff, yoff, zoff = grid_offset

    xcoords = torch.arange(0., width, x_grid_res) + xoff
    ycoords = torch.arange(0., height, y_grid_res) + yoff
    zcoords = torch.arange(0., depth, z_grid_res) + zoff

    zz, yy, xx = torch.meshgrid(zcoords, ycoords, xcoords)
    grid = torch.stack([xx, yy, zz], dim=-1)
    grid = grid.permute(2,1,0,3)
    voxel_corners = torch.cat((grid[:-1, :-1, :-1].unsqueeze(-1),
                               grid[:-1, :-1, 1:].unsqueeze(-1),
                               grid[:-1, 1:, :-1].unsqueeze(-1),
                               grid[:-1, 1:, 1:].unsqueeze(-1),
                               grid[1:, :-1, :-1].unsqueeze(-1),
                               grid[1:, :-1, 1:].unsqueeze(-1),
                               grid[1:, 1:, :-1].unsqueeze(-1),
                               grid[1:, 1:, 1:].unsqueeze(-1)), dim=-1)  
    voxel_pos = torch.mean(voxel_corners, dim=-1)

    return grid, voxel_pos, voxel_corners


def prepare_targets(targets):

    heatmaps = torch.stack([t.get_field("hm") for t in targets])
    regression = torch.stack([t.get_field("reg") for t in targets])

    cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
    p_ints = torch.stack([t.get_field("p_ints") for t in targets])
    p_ints_voxel = torch.stack([t.get_field("p_ints_voxel") for t in targets])
    p_offsets = torch.stack([t.get_field("p_offsets") for t in targets])
    dimensions = torch.stack([t.get_field("dimensions") for t in targets])
    locations = torch.stack([t.get_field("locations") for t in targets])
    orientation = torch.stack([t.get_field("orientation") for t in targets])
    reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
    voxel = torch.stack([t.get_field("voxel") for t in targets])
    sdf_feat = torch.stack([t.get_field("sdf_feat") for t in targets])
    sdf_mask = torch.stack([t.get_field("sdf_mask") for t in targets])
    sdf_idx = torch.stack([t.get_field("sdf_idx") for t in targets])
    sdf = torch.stack([t.get_field("sdf") for t in targets])

    return heatmaps, regression, voxel, dict(cls_ids=cls_ids,
                                             p_ints=p_ints,
                                             p_ints_voxel=p_ints_voxel,
                                             p_offsets=p_offsets,
                                             dimensions=dimensions,
                                             locations=locations,
                                             orientation=orientation,
                                             reg_mask=reg_mask,
                                             sdf_feat=sdf_feat,
                                             sdf_mask=sdf_mask,
                                             sdf_idx=sdf_idx,
                                             sdf=sdf)


def prepare_predictions(dim_reference, grid_size, targets_variables, pred_regression):

    # #
    batch, channel = pred_regression.shape[0], pred_regression.shape[1]
    targets_p_ints_voxel = targets_variables["p_ints_voxel"]
    targets_p_ints = targets_variables["p_ints"]
    targets_p_ints = targets_p_ints.view(-1, targets_p_ints.shape[2])
    device = pred_regression.device
    dim_reference = dim_reference.to(device)

    # # obtain prediction from points of interests
    pred_regression_pois = utils_encode.select_point_of_interest(batch, targets_p_ints_voxel, pred_regression)
    pred_regression_pois = pred_regression_pois.view(-1, channel)

    pred_p_offsets = pred_regression_pois[:, :3]
    pred_dimensions_offsets = pred_regression_pois[:, 3:6]
    pred_orientation = pred_regression_pois[:, 6:]

    # #
    pred_dimensions = utils_encode.decode_dimension(
        targets_variables["cls_ids"],
        pred_dimensions_offsets,
        dim_reference
    )
    pred_locations = utils_encode.decode_location(
        grid_size,
        targets_p_ints,
        pred_p_offsets
        )

    # #
    pred_box3d_orientation = utils_encode.encode_box3d(
        pred_orientation,
        targets_variables["dimensions"],
        targets_variables["locations"]
    )
    pred_box3d_dims = utils_encode.encode_box3d(
        targets_variables["orientation"],
        pred_dimensions,
        targets_variables["locations"]
    )
    pred_box3d_locs = utils_encode.encode_box3d(
        targets_variables["orientation"],
        targets_variables["dimensions"],
        pred_locations
    )

    return dict(ori_box=pred_box3d_orientation,
                dim_box=pred_box3d_dims,
                loc_box=pred_box3d_locs, 
            )