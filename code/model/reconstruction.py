
import torch
from torch import nn
from torch.nn import functional as F

class reconstruction(nn.Module):
    def __init__(
        self,
        in_channels,
        head_conv,
        feat_dim,
    ):
        super().__init__()

        ## voxel
        self.voxel_head = nn.Sequential(
            nn.Conv1d(in_channels,head_conv,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_conv,1,1)
        ) 

        ## regression
        self.sdf_head = nn.Sequential(
            nn.Conv1d(in_channels,head_conv,1),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_conv,feat_dim,1)
        ) 

    def forward(self, features):

        # #
        B, feat_dim, W, H, D = features.shape
        features = features.view(B, feat_dim, W*H*D)

        voxel = self.voxel_head(features)
        sdf_feat = self.sdf_head(features)

        # 
        voxel = torch.sigmoid(voxel)

        return voxel.permute(0,2,1), sdf_feat.permute(0,2,1)