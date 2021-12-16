
import math
import torch
from torch import nn


def group_norm(out_channels):
    num_groups = 32
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)


def _fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def sigmoid_hm(hm_features):
    x = hm_features.sigmoid_()
    x = x.clamp(min=1e-4, max=1 - 1e-4)

    return x

def get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:1])
        e = sum(reg_channels[:2])
    elif name == "ori":
        s = sum(reg_channels[:2])
        e = sum(reg_channels)

    return slice(s, e, 1)