
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model.resnet import *
from model.featlift import *
from model.detector import *
from model.reconstruction import *
import utils.general as utils_general

class JointDetRecon(nn.Module):
    def __init__(self, conf):
        super().__init__()

        dataset_conf = conf.get_config("dataset")
        voxel_size = dataset_conf.get_float("voxel_size")
        grid_size = dataset_conf.get_list("grid_size")
        grid_offset = dataset_conf.get_list("grid_offset")
        img_res = dataset_conf.get_list("img_res")
        input_width = img_res[1]
        input_height = img_res[0]

        grid_res = (voxel_size, voxel_size, voxel_size)
        grid_range = (grid_size[0]+voxel_size, grid_size[1]+voxel_size, grid_size[2]+voxel_size)
        _, grid_pos, _ = utils_general.make_grid(grid_range, grid_res, grid_offset)

        self.backbone_network = resnet34(pretrained=conf.get('model.backbone_network.is_pretrain'))
        self.lifting_network = featlifting(grid_pos, input_width, input_height, **conf.get_config('model.lifting_network'))
        self.detector_network = detector(**conf.get_config('model.detector_network'))
        self.recon_network = reconstruction(**conf.get_config('model.recon_network'))

    def forward(self, K, RT, images):

        #
        feat8, feat16, feat32 = self.backbone_network.forward(images)
        #
        feat_3d, visible = self.lifting_network.forward(K, RT, feat8, feat16, feat32)
        #
        predict_class, predict_regression = self.detector_network.forward(feat_3d)
        #
        predict_voxel, predict_sdf_feat = self.recon_network.forward(feat_3d)

        output = {
            "predict_class": predict_class,
            "predict_regression": predict_regression,
            "predict_voxel": predict_voxel,
            "predict_sdf_feat": predict_sdf_feat,
            "visible": visible
        }

        return output
