
import numpy as np
import torch
PI = np.pi


def affine_transform(point, matrix):
    point_exd = np.array([point[0], point[1], 1.])
    new_point = np.matmul(matrix, point_exd)

    return new_point[:2]

def rotz(orientation_1, orientation_2):
    """Rotation about the z-axis."""
    c = orientation_1
    s = orientation_2
    return np.array([[s,  0, -c],
                     [0,  1,  0],
                     [c,  0,  s]])

def encode_label(dims, locs, orientation_1, orientation_2):

    R = rotz(orientation_1, orientation_2)
    l, w, h = dims[0], dims[1], dims[2]

    x_corners = [-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2]
    y_corners = [w/2,w/2,w/2,w/2,-w/2,-w/2,-w/2,-w/2]
    z_corners = [h/2,h/2,-h/2,-h/2,h/2,h/2,-h/2,-h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += locs[0]
    corners_3d[1,:] += locs[1]
    corners_3d[2,:] += locs[2]

    return corners_3d.astype(np.float32), locs.astype(np.float32)


def select_point_of_interest(batch, index, feature_maps):
    '''
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format [B, 30, 1]
        feature_maps: regression feature map in [B, C, W, H, D]

    Returns:

    '''
    B, C, W, H, D = feature_maps.shape
    feature_maps = feature_maps.view(B, C, W*H*D).permute(0, 2, 1).contiguous()
    index = index.repeat(1, 1, C)

    # select specific features bases on POIs
    feature_maps = feature_maps.gather(1, index.long())

    return feature_maps


def decode_location(grid_size, center_int, center_offset):
    '''
    '''
    center = center_int + center_offset*(grid_size/2)

    return center

def decode_dimension(cls_id, dims_offset, dim_ref):
    '''
    retrieve object dimensions
    Args:
        cls_id: each object id
        dims_offset: dimension offsets, shape = (N, 3)

    Returns:

    '''
    cls_id = cls_id.flatten().long()

    dims_select = dim_ref[cls_id,:]
    dimensions = dims_offset.exp() * dims_select

    return dimensions


def rad_to_matrix(oris, N):
    device = oris.device

    cos = oris[:,0]
    sin = oris[:,1]

    i_temp = torch.tensor([[1,  0, -1],
                           [0,  1, 0],
                           [1,  0, 1]]).to(dtype=torch.float32,
                                           device=device)
    ry = i_temp.repeat(N, 1).view(N, -1, 3)

    ry[:, 0, 0] *= sin
    ry[:, 0, 2] *= cos
    ry[:, 2, 0] *= cos
    ry[:, 2, 2] *= sin

    return ry

def rotz(orientation_1, orientation_2):
    """Rotation about the z-axis."""
    c = orientation_1
    s = orientation_2
    return np.array([[s,  0, -c],
                     [0,  1,  0],
                     [c,  0,  s]])

def encode_label(dims, locs, orientation_1, orientation_2):

    R = rotz(orientation_1, orientation_2)
    l, w, h = dims[0], dims[1], dims[2]

    x_corners = [-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2]
    y_corners = [w/2,w/2,w/2,w/2,-w/2,-w/2,-w/2,-w/2]
    z_corners = [h/2,h/2,-h/2,-h/2,h/2,h/2,-h/2,-h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += locs[0]
    corners_3d[1,:] += locs[1]
    corners_3d[2,:] += locs[2]

    return corners_3d.astype(np.float32), locs.astype(np.float32)



def encode_box3d(oris, dims, locs):
    '''
    construct 3d bounding box for each object.
    Args:
        oris: orientations of objects  
        dims: dimensions of objects    
        locs: locations of objects    

    Returns:

    '''
    if len(oris.shape) == 3:
        oris = oris.view(-1, 2)
    if len(dims.shape) == 3:
        dims = dims.view(-1, 3)
    if len(locs.shape) == 3:
        locs = locs.view(-1, 3)
    device = oris.device
    N = oris.shape[0]

    R = rad_to_matrix(oris, N) 

    dims = dims.view(-1, 1).repeat(1, 8)*0.5
    dims[::3, [0,3,4,7]] = -1*dims[::3, [0,3,4,7]]
    dims[1::3, [4,5,6,7]] = -1*dims[1::3, [4,5,6,7]]
    dims[2::3, [2,3,6,7]] = -1*dims[2::3, [2,3,6,7]]

    box_3d = torch.matmul(R, dims.view(N, 3, -1))
    box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

    return box_3d
