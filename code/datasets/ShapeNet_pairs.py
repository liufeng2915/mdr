
import os
import csv
import random
import numpy as np
from PIL import Image,ImageOps
import scipy.spatial
import scipy.io
from torch.utils.data import Dataset
import utils.general as utils_general
import utils.transforms as utils_transforms
import utils.data as utils_data
import utils.encode as utils_encode
import utils.gaussian as utils_gaussian

TYPE_ID_CONVERSION = {
"bed":0, 
"bottle":1, 
"bowl":2, 
"car":3, 
"chair":4, 
"display":5, 
"guitar":6, 
"lamp":7, 
"motorbike":8, 
"mug":9,
"piano":10,
"pillow":11,
"sofa":12,
"table":13,
}

class ShapeNetPairsDataset(Dataset):
    def __init__(self, 
                 is_train,
                 pca_data,
                 data_dir,
                 img_res,
                 voxel_size,
                 grid_size,
                 grid_offset,
                 max_objs,
                 detect_classes,
                 dim_reference,
                 num_train_voxel_sdf,
                 pixel_mean,
                 pixel_std,
                 ):

        ## data folder
        self.data_root = data_dir
        self.data_list_path = os.path.join(self.data_root, "data_list.txt")

        self.is_train = is_train

        self.image_dir = os.path.join(self.data_root, "image")
        self.label_dir = os.path.join(self.data_root, "label")
        self.calib_dir = os.path.join(self.data_root, "calib")
        self.voxel_dir = os.path.join(self.data_root, "voxel_samples")

        ## pca model
        self.pca_base = pca_data["pca_base"]
        self.pinv_pca_base = np.linalg.pinv(self.pca_base)
        self.feat_dim = self.pca_base.shape[0]
        self.pca_mean = pca_data["pca_mean"]
        self.mean_latent = pca_data["mean_latent"]
        self.std_latent = pca_data["std_latent"]

        ## sample list
        fid = open(self.data_list_path)
        self.sample_files = fid.read().splitlines()
        fid.close()
        self.num_samples = len(self.sample_files)

        ## 
        self.num_train_voxel_sdf = num_train_voxel_sdf
        self.max_objs = max_objs
        self.classes = detect_classes
        self.num_classes = len(self.classes)
        self.input_width = img_res[1]
        self.input_height = img_res[0]

        ##  3D space 
        grid_res = (voxel_size, voxel_size, voxel_size)
        grid_range = (grid_size[0]+voxel_size, grid_size[1]+voxel_size, grid_size[2]+voxel_size)
        grid, grid_pos, voxel_corners = utils_general.make_grid(grid_range, grid_res, grid_offset)
        self.voxel_width, self.voxel_height, self.voxel_depth, point_dim = grid_pos.size()
        self.grid_pos = grid_pos.view(self.voxel_width*self.voxel_height*self.voxel_depth,point_dim).numpy()

        self.grid_radius = (3*(voxel_size**2))**(1/2)/2
        self.grid = grid.reshape(grid.shape[0]*grid.shape[1]*grid.shape[2], 3).numpy()

        if self.is_train:
            self.transforms = utils_transforms.build_transforms(pixel_mean, pixel_std)
        else:
            self.transforms = utils_transforms.build_transforms(pixel_mean, pixel_std)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        # # image
        file_name = self.sample_files[idx]
        img_path = os.path.join(self.image_dir, file_name + '_pbrt.png')
        img = Image.open(img_path)

        # # calib
        calib_path = os.path.join(self.calib_dir, file_name + '.txt')
        RT, K = self.readCalibration(calib_path)

        # # label
        label_path = os.path.join(self.label_dir, file_name + '.txt')
        anns = self.readLabels(label_path)

        # # voxel and local sdf
        voxel_path = os.path.join(self.voxel_dir, file_name + '.mat')
        voxel, train_sdf, train_sdf_mask, train_sdf_idx, train_sdf_feat = self.readVoxel(voxel_path)

        #
        file_name = file_name.replace("/","_")

        # # 
        if not self.is_train:
            target = utils_data.ParamsList(image_size=img.size,
                                           is_train=self.is_train)
            target.add_field("RT", RT.astype(np.float32))
            target.add_field("K", K.astype(np.float32))

            if self.transforms is not None:
                file_name, img, target = self.transforms(file_name, img, target)
            return file_name, img, target

        # # 
        heat_map = np.zeros([self.num_classes, self.voxel_width, self.voxel_height, self.voxel_depth], dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        p_ints = np.zeros([self.max_objs, 3], dtype=np.float32)
        p_ints_voxel = np.zeros([self.max_objs, 1], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 3], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        orientation = np.zeros([self.max_objs, 2], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)

        # #
        for i, a in enumerate(anns):
            a = a.copy()
            cls = a["label"]

            # # encode to 3D boxes
            box3d, center_3d = utils_encode.encode_label(a["dimensions"], a["centroid"], a["orientation_1"], a["orientation_2"])

            # # 3D heatmap
            if (np.min(self.grid[:,0]) <= center_3d[0] <= np.max(self.grid[:,0])) and \
               (np.min(self.grid[:,1]) <= center_3d[1] <= np.max(self.grid[:,1])) and \
               (np.min(self.grid[:,2]) <= center_3d[2] <= np.max(self.grid[:,2])):

                _, center_grid_idx = scipy.spatial.KDTree(self.grid_pos).query(np.expand_dims(center_3d, axis=0))
                c_idx,c_idy,c_idz = np.unravel_index(center_grid_idx,(self.voxel_width, self.voxel_height, self.voxel_depth))
                center_3d_int = np.concatenate((c_idx,c_idy,c_idz),axis=0)
                p_offset = center_3d - self.grid_pos[center_grid_idx]
                radius = utils_gaussian.gaussian_radius(a["dimensions"][0], a["dimensions"][1], a["dimensions"][2])
                radius = max(0, 2*int(radius/self.grid_radius))
                #if radius == 0:
                #    radius = 1
                #print(center_3d_int, radius)
                heat_map[cls] = utils_gaussian.draw_umich_gaussian(heat_map[cls], center_3d_int, radius)

                cls_ids[i] = cls
                regression[i] = box3d
                p_ints[i] = self.grid_pos[center_grid_idx]
                p_ints_voxel[i] = center_grid_idx
                p_offsets[i] = p_offset
                dimensions[i] = np.array(a["dimensions"])
                locations[i] = a["centroid"]
                orientation[i,0] = a["orientation_1"]
                orientation[i,1] = a["orientation_2"]
                reg_mask[i] = 1


        target = utils_data.ParamsList(image_size=img.size,
                                       is_train=self.is_train)
        target.add_field("hm", heat_map)
        target.add_field("reg", regression)
        target.add_field("cls_ids", cls_ids)
        target.add_field("p_ints", p_ints)
        target.add_field("p_ints_voxel", p_ints_voxel)
        target.add_field("p_offsets", p_offsets)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("orientation", orientation)
        target.add_field("K", K.astype(np.float32))
        target.add_field("RT", RT.astype(np.float32))
        target.add_field("voxel", voxel.astype(np.float32))
        target.add_field("sdf_feat", train_sdf_feat.astype(np.float32))
        target.add_field("sdf_mask", train_sdf_mask.astype(np.float32))
        target.add_field("sdf_idx", train_sdf_idx.astype(np.float32))
        target.add_field("sdf", train_sdf.astype(np.float32))
        target.add_field("reg_mask", reg_mask)

        if self.transforms is not None:
            file_name, img, target = self.transforms(file_name, img, target)

        return file_name, img, target


    # # ------------------------------------------------------------------------------------------ # #
    # # read label
    def readLabels(self, path):
        annotations = []
        lines = [line.rstrip() for line in open(path)]
        for line in lines:
            data = line.split(' ')
            data[1:] = [float(x) for x in data[1:]]
            classname = data[0]
            if classname in self.classes:
                annotations.append({
                    "class": classname,
                    "label": TYPE_ID_CONVERSION[classname],
                    "box2d": np.array([data[1], data[2], data[3], data[4]]).astype(np.float32),
                    "centroid": np.array([data[5],data[6],data[7]]).astype(np.float32),
                    "dimensions": np.array([data[8],data[9],data[10]]).astype(np.float32), #[l,w,h]
                    "orientation_1": np.array(data[11]).astype(np.float32),
                    "orientation_2": np.array(data[12]).astype(np.float32),
                })

        return annotations

    # # read calib
    def readCalibration(self, path):
        lines = [line.rstrip() for line in open(path)]
        RT = np.array([float(x) for x in lines[0].split(' ')])
        RT = np.reshape(RT, (3,3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        K = np.reshape(K, (3,3), order='F')

        return RT, K

    # # read local SDF
    def readVoxel(self, path):
        mat_file = scipy.io.loadmat(path)
        voxel = mat_file['voxel']
        sdf_data = mat_file['sdf_data']
        sdf_idx = np.int32(sdf_data[:,0])
        sdf = sdf_data[:,1:]

        #
        p_num_voxel_sdf = self.num_train_voxel_sdf
        train_sdf = np.zeros((p_num_voxel_sdf, sdf.shape[1]))
        train_sdf_idx = np.zeros((p_num_voxel_sdf)).astype(np.int32)
        train_sdf_mask = np.zeros((p_num_voxel_sdf))
        if len(sdf_idx)<p_num_voxel_sdf:
            rand_idx = np.random.choice(len(sdf_idx), len(sdf_idx))
        else:
            rand_idx = np.random.choice(len(sdf_idx), p_num_voxel_sdf)
        train_sdf[:len(rand_idx)] = sdf[rand_idx]
        train_sdf_mask[:len(rand_idx)] = 1
        train_sdf_idx[:len(rand_idx)] = sdf_idx[rand_idx]

        train_sdf_feat = np.zeros((p_num_voxel_sdf, self.feat_dim))
        offset = train_sdf[:len(rand_idx)] - self.pca_mean
        latent = np.matmul(offset, self.pinv_pca_base)
        train_sdf_feat[:len(rand_idx)] = latent

        #
        voxel[sdf_idx] = 1
        return voxel, train_sdf, train_sdf_mask, train_sdf_idx, train_sdf_feat 
