
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as torch_F
from PIL import Image

def nms_hm(heat_map, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch_F.max_pool3d(heat_map,
                        kernel_size=(kernel, kernel, kernel),
                        stride=1,
                        padding=pad)
    eq_index = (hmax == heat_map).float()

    return heat_map * eq_index


def visualize_image(image, dataset_conf, save_path):

    pixel_mean = dataset_conf.get("pixel_mean")
    pixel_std = dataset_conf.get("pixel_std")

    image = image.cpu().detach()
    image = F.normalize(image, mean=-torch.as_tensor(pixel_mean), std=1./torch.as_tensor(pixel_std))
    image = image[[2, 1, 0]]
    image = (image+1)/2
    image = image.permute(1,2,0)

    image = Image.fromarray((image.numpy()*255).astype(np.uint8)).convert("L")
    image.save('{0}/image.png'.format(save_path))

    return 1
    

def visualize_hm(heatmaps, visible, save_path, type):

    hm = heatmaps.cpu().detach()
    visible = visible.cpu()
    hm = hm*visible
    
    hm_values, indices = torch.max(hm, 0)
    values, indices = torch.max(hm_values, 1)
    values = values.T.numpy()
    values = np.flip(values, 0)

    cm = plt.get_cmap('jet')
    colored_hm_image = cm(values)
    colored_hm_image = Image.fromarray((colored_hm_image[:, :, :3] * 255).astype(np.uint8))
    colored_hm_image.save('{0}/heatmap_{1}.png'.format(save_path, type))

    return 


def visualize_voxe(heatmaps, visible, save_path, type):

    _, W, H, D = visible.shape
    hm = heatmaps.cpu().detach()
    hm = hm.view(W,H,D,1).permute(3,0,1,2)
    visible = visible.cpu()
    hm = hm*visible
    
    hm = torch.squeeze(hm)
    values, indices = torch.max(hm, 1)
    values = values.T.numpy()
    values = np.flip(values, 0)

    cm = plt.get_cmap('jet')
    colored_hm_image = cm(values)
    colored_hm_image = Image.fromarray((colored_hm_image[:, :, :3] * 255).astype(np.uint8))
    colored_hm_image.save('{0}/voxel_{1}.png'.format(save_path, type))

    return 