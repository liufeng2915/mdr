
import numpy as np
PI = np.pi

def gaussian_radius(l, h, w, thresh_min=0.7):

    t = thresh_min
    a = 4/3.*PI
    b = t*l*h*w
    r = (b/a)**(1/3)

    return r

def gaussian(shape, sigma=1):
    m, n, o = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m + 1, -n:n + 1, -o:o + 1]

    h = np.exp(-(x * x + (y * y)  + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian((diameter, diameter, diameter), sigma=diameter / 6)

    x, y, z = center[0], center[1], center[2]

    width, height, depth = heatmap.shape[0:3]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    far, near = min(z, radius), min(depth - z, radius + 1)

    masked_heatmap = heatmap[x - left:x + right, y - top:y + bottom, z - far:z + near]
    masked_gaussian = gaussian[radius - left:radius + right, radius - top:radius + bottom, radius - far:radius + near]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap
