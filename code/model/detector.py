
import torch
from torch import nn
from torch.nn import functional as F
from model.det_head_layers import (
    sigmoid_hm,
    _fill_fc_weights,
    group_norm
)


class detector(nn.Module):
    def __init__(
        self,
        in_channels,
        head_conv,
        num_classes,
        num_regression,
        regression_channels
    ):
        super().__init__()

        assert sum(regression_channels) == num_regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                regression_channels, num_regression)

        ## hm
        self.class_head = nn.Sequential(
            nn.Conv3d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),
            
            group_norm(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv3d(head_conv,
                      num_classes,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        ) 
        self.class_head[-1].bias.data.fill_(-2.19)

        ## regression
        self.regression_head = nn.Sequential(
            nn.Conv3d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            group_norm(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv3d(head_conv,
                      num_regression,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        _fill_fc_weights(self.regression_head)


    def forward(self, features):

        # #
        head_class = self.class_head(features)
        head_regression = self.regression_head(features)

        # 
        head_class = sigmoid_hm(head_class)
        #
        offset_p = head_regression[:, :3, ...].clone() 
        head_regression[:, :3, ...] = torch.tanh(offset_p)
        #
        offset_dims = head_regression[:, 3:6, ...].clone()
        head_regression[:, 3:6, ...] = torch.sigmoid(offset_dims) - 0.5
        #
        vector_ori = head_regression[:, 6:, ...].clone()
        head_regression[:, 6:, ...] = F.normalize(vector_ori)

        return head_class, head_regression