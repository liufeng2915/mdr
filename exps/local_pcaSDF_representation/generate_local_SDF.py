

import os
import scipy.io
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import glob
import torch
import scipy.spatial

def get_sdf(sdf_file, sdf_res):
    intsize = 4
    floatsize = 8
    sdf = {
        "param": [],
        "value": []
    }
    with open(sdf_file, "rb") as f:
        try:
            bytes = f.read()
            ress = np.fromstring(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != sdf_res or ress[1] != sdf_res or ress[2] != sdf_res:
                raise Exception(sdf_file, "res not consistent with ", str(sdf_res))
            positions = np.fromstring(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["param"] = [positions[0], positions[1], positions[2],
                            positions[3], positions[4], positions[5]]
            sdf["param"] = np.float32(sdf["param"])
            sdf["value"] = np.fromstring(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["value"] = np.reshape(sdf["value"], (sdf_res + 1, sdf_res + 1, sdf_res + 1))
        finally:
            f.close()
    return sdf


def make_grid(grid_size, grid_res, grid_offset):

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


def make_voxel_sample(sample_step, voxel_size):

	coords = torch.arange(0., voxel_size*2, sample_step) #  2*radius
	zz, yy, xx = torch.meshgrid(coords, coords, coords)
	grid = torch.stack([xx, yy, zz], dim=-1)
	voxel_pts = np.reshape(grid.numpy(), (grid.shape[0]*grid.shape[1]*grid.shape[2], 3)) 
	voxel_pts = voxel_pts[:,[1,0,2]]
	voxel_pts = voxel_pts - np.mean(voxel_pts, axis=0)

	return voxel_pts

##
sdf_res = 256        #128 256 64
sdf_thres = 0.005
voxel_size = 0.035
sample_step = 0.005
grid_res = (voxel_size, voxel_size, voxel_size)
grid_size = (1.12+voxel_size, 1.12+voxel_size, 1.12+voxel_size)
grid_offset = (-0.56, -0.56, -0.56)

##
grid, voxel_pos, voxel_corners = make_grid(grid_size, grid_res, grid_offset)
w,h,l,_ = voxel_pos.shape
voxel_pos = voxel_pos.view(w*h*l, 3).data.numpy()
grid = grid.reshape((w+1)*(h+1)*(l+1), 3).data.numpy()
voxel_corners = voxel_corners.view(w*h*l, 3, 8).data.numpy()
kdt = scipy.spatial.KDTree(voxel_pos)
#scipy.io.savemat('voxel_pos.mat', {'voxel_pos': voxel_pos, 'voxel_corners':voxel_corners})

##
voxel_pts = make_voxel_sample(sample_step, voxel_size)
#scipy.io.savemat('voxel_pts.mat', {'voxel_pts': voxel_pts})

##
file_path = 'test.obj'
dist_name = curr_list[k].split('/')[-1][:-4]+'.dist'
command_str1 = 'isosurface/computeDistanceField ' + file_path + ' ' + str(sdf_res) + ' ' + str(sdf_res) + ' ' + str(sdf_res) + ' -s  -e 1.2 -o ' + dist_name + ' -m 1'
os.system(command_str1)

try:
	sdf_dict = get_sdf(dist_name, sdf_res)
	params = sdf_dict["param"]
	sdf_values = sdf_dict["value"]
	x_ = np.linspace(params[0], params[3], num=sdf_res + 1).astype(np.float32)
	y_ = np.linspace(params[1], params[4], num=sdf_res + 1).astype(np.float32)
	z_ = np.linspace(params[2], params[5], num=sdf_res + 1).astype(np.float32)
	x,y,z = np.meshgrid(z_, y_, x_, indexing='ij')
	x = np.expand_dims(x, 3)
	y = np.expand_dims(y, 3)
	z = np.expand_dims(z, 3)
	points = np.concatenate((x, y, z), axis=3)
	points = np.reshape(points, ((sdf_res+1)**3,3))
	points = points[:,[2,1,0]]
	sdf_values = np.reshape(sdf_values, ((sdf_res+1)**3,1))
	points_kdt = scipy.spatial.KDTree(points)

	# #
	sdf_expand = 10
	valid_pts = points[np.where(sdf_values<sdf_thres*sdf_expand)[0],:]
	valid_sdf = sdf_values[np.where(sdf_values<sdf_thres*sdf_expand)[0],:]

	_, temp_idx = kdt.query(np.expand_dims(valid_pts, axis=0))
	u_temp_idx = np.unique(temp_idx)
	voxel = np.zeros((voxel_pos.shape[0],1))
	check_idx = []
	for ik in range(len(u_temp_idx)):
		t_index = np.where(u_temp_idx[ik]==temp_idx)[1]
		t_sdf = valid_sdf[t_index, :]

		temp_sta = len(np.where(t_sdf<=sdf_thres)[1])/(len(t_sdf)+1e-8)
		voxel[u_temp_idx[ik]] = temp_sta
		if temp_sta>0:
			check_idx = np.append(check_idx, ik)
	u_temp_idx = u_temp_idx[np.int64(check_idx)]

	# #
	valid_pos = voxel_pos[u_temp_idx, :]
	valid_pos = np.reshape(valid_pos, (1, valid_pos.shape[0], valid_pos.shape[1]))
	valid_pos = np.repeat(valid_pos, voxel_pts.shape[0], axis=0)
	valid_pos = np.reshape(np.transpose(valid_pos, (1, 0, 2)), (valid_pos.shape[0]*valid_pos.shape[1], valid_pos.shape[2]))
	temp_cell_pts = np.reshape(voxel_pts, (1, voxel_pts.shape[0], voxel_pts.shape[1]))
	temp_cell_pts = np.repeat(temp_cell_pts, len(u_temp_idx), axis=0)
	temp_cell_pts = np.reshape(temp_cell_pts, (temp_cell_pts.shape[0]*temp_cell_pts.shape[1], temp_cell_pts.shape[2]))
	whole_pts = valid_pos+temp_cell_pts

	_, whole_idx = points_kdt.query(np.expand_dims(whole_pts, axis=0))
	whole_sdf = sdf_values[whole_idx,:]
	whole_sdf = np.squeeze(whole_sdf, axis=0)
	sdf_data = np.reshape(whole_sdf, (len(u_temp_idx), voxel_pts.shape[0]))
	sdf_data = np.concatenate((np.reshape(u_temp_idx, (len(u_temp_idx), 1)), sdf_data), axis=1)

	scipy.io.savemat('test.mat', {'sdf_data': sdf_data.astype(np.float32), 'voxel':voxel.astype(np.float32)})
	os.remove(dist_name)
except:
	continue
