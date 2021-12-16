
import torch
import numpy as np
from collections import defaultdict
from skimage import transform as trans

class ParamsList():
    """
    This class represents labels of specific object.
    """

    def __init__(self, image_size, is_train=True):
        self.size = image_size
        self.is_train = is_train
        self.extra_fields = {}

    def add_field(self, field, field_data):
        field_data = field_data if isinstance(field_data, torch.Tensor) else torch.as_tensor(field_data)
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, target):
        for k, v in target.extra_fields.items():
            self.extra_fields[k] = v

    def to(self, device):
        target = ParamsList(self.size, self.is_train)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            target.add_field(k, v)
        return target

    def __len__(self):
        if self.is_train:
            reg_num = len(torch.nonzero(self.extra_fields["reg_mask"]))
        else:
            reg_num = 0
        return reg_num

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "regress_number={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


class ImageList():
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def collator(batch):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """
    transposed_batch = list(zip(*batch))
    images = to_image_list(transposed_batch[1])
    targets = transposed_batch[2]
    img_name = transposed_batch[0]
    return dict(images=images,
                targets=targets,
                img_name=img_name)
