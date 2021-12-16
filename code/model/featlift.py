
import torch
from torch import nn
from torch.nn import functional as F
from model.unet3d import *

class featlifting(nn.Module):
    def __init__(
        self,
        grid_pos,
        img_width,
        img_height,
        num_levels,
        d_in,
        is_positional_encoding
    ):
        super().__init__()

        self.grid_pos = grid_pos
        self.img_width = img_width
        self.img_height = img_height
        self.is_positional_encoding = is_positional_encoding
        self.use_visible = True

        if self.is_positional_encoding:
            self.pre_conv = nn.Conv3d(64, d_in-3, kernel_size=1)
        else:
            self.pre_conv = nn.Conv3d(64, d_in, kernel_size=1)
        self.unet = UNet3D(in_channels=d_in, out_channels=d_in, final_sigmoid=False, num_levels=num_levels)

    def perspective(self, K, RT, vector):
        proj_matrix = torch.bmm(K, RT)
        proj_matrix = proj_matrix.view(-1,1,1,1,3,3)
        vector = vector.unsqueeze(-1) 
        homogenous = torch.matmul(proj_matrix, vector)
        homogenous = homogenous.squeeze(-1)

        return homogenous[..., :-1] / homogenous[..., [-1]]


    def forward(self, K, RT, feat8, feat16, feat32):

        batch_size = K.shape[0]
        device = K.device
        grid_pos = self.grid_pos.clone().repeat(batch_size, 1, 1, 1, 1).to(device)
        grid_pos = grid_pos.permute(0,2,1,3,4)
        img_pos = self.perspective(K, RT, grid_pos)
        img_size = grid_pos.new([self.img_width, self.img_height])
        # # visible info
        visible = ((img_pos[:,:,:,:,0]>0) & (img_pos[:,:,:,:,0]<img_size[0]) & (img_pos[:,:,:,:,1]>0) & (img_pos[:,:,:,:,1]<img_size[1])).float()
        visible = torch.unsqueeze(visible, 1)
        norm_img_pos = (2 * img_pos / img_size -1).clamp(-1, 1)
        norm_img_pos = norm_img_pos.flatten(2, 3) 

        # #
        feat8_3d = F.grid_sample(feat8, norm_img_pos, align_corners=True)
        feat16_3d = F.grid_sample(feat16, norm_img_pos, align_corners=True)
        feat32_3d = F.grid_sample(feat32, norm_img_pos, align_corners=True)
        feat_3d = feat8_3d + feat16_3d + feat32_3d
        feat_3d = feat_3d.view(feat_3d.shape[0], feat_3d.shape[1], self.grid_pos.shape[1], self.grid_pos.shape[0], self.grid_pos.shape[2])
        feat_3d = feat_3d.permute(0,1,3,2,4)
        visible = visible.permute(0,1,3,2,4)
        feat_3d = self.pre_conv(feat_3d)

        if self.use_visible:
            feat_3d = feat_3d * visible

        # #
        if self.is_positional_encoding:
            feat_3d = torch.cat((feat_3d, self.grid_pos.clone().repeat(batch_size, 1, 1, 1, 1).to(device).permute(0,4,1,2,3)), dim=1)
        feat_3d = self.unet(feat_3d)

        return feat_3d, visible.detach()