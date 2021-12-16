
import torch
from torchvision.transforms import functional as F

import numpy as np
from numpy import random
import cv2
import math
import os
import sys

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, file_name, image, target):
        for t in self.transforms:
            file_name, image, target = t(file_name, image, target)
        return file_name, image, target


class ToTensor():
    def __call__(self, file_name, image, target):
        image = F.to_tensor(image)
        if torch.max(image)>1:
            image = image/255.0
        return file_name, image, target

class ConvertToFloat(object):
    """
    Converts image data type to float.
    """
    def __call__(self, file_name, image, imobj=None):
        image = np.asarray(image)
        return file_name, image.astype(np.float32), imobj

class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, file_name, image, target):
        if self.to_bgr:
            image = image[[2, 1, 0]]
        image = F.normalize(image, mean=self.mean, std=self.std)
        return file_name, image, target


class RandomContrast(object):
    """
    Randomly adjust contrast of an image given lower and upper bound,
    and a distortion probability.
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.lower = lower
        self.upper = upper
        self.distort_prob = distort_prob

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, file_name, image, imobj=None):
        if random.rand() <= self.distort_prob:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return file_name, image, imobj

class RandomSaturation(object):
    """
    Randomly adjust the saturation of an image given a lower and upper bound,
    and a distortion probability.
    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.distort_prob = distort_prob
        self.lower = lower
        self.upper = upper

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, file_name, image, imobj=None):
        if random.rand() <= self.distort_prob:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return file_name, image, imobj


class RandomHue(object):
    """
    Randomly adjust the hue of an image given a delta degree to rotate by,
    and a distortion probability.
    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, file_name, image, imobj=None):
        if random.rand() <= self.distort_prob:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return file_name, image, imobj


class ConvertColor(object):
    """
    Converts color spaces to/from HSV and BGR
    """
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, file_name, image, imobj=None):

        # BGR --> HSV
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV --> BGR
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        else:
            raise NotImplementedError

        return file_name, image, imobj


class RandomBrightness(object):
    """
    Randomly adjust the brightness of an image given given a +- delta range,
    and a distortion probability.
    """
    def __init__(self, distort_prob, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, file_name, image, imobj=None):
        if random.rand() <= self.distort_prob:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return file_name, image, imobj


class PhotometricDistort(object):
    """
    Packages all photometric distortions into a single transform.
    """
    def __init__(self, distort_prob):

        self.distort_prob = distort_prob

        # contrast is duplicated because it may happen before or after
        # the other transforms with equal probability.
        self.transforms = [
            RandomContrast(distort_prob),
            ConvertColor(transform='HSV'),
            RandomSaturation(distort_prob),
            RandomHue(distort_prob),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(distort_prob)
        ]

        self.rand_brightness = RandomBrightness(distort_prob)

    def __call__(self, file_name, image, imobj):

        # do contrast first
        if random.rand() <= 0.5:
            distortion = self.transforms[:-1]

        # do contrast last
        else:
            distortion = self.transforms[1:]

        # add random brightness
        distortion.insert(0, self.rand_brightness)

        # compose transformation
        distortion = Compose(distortion)

        return distortion(file_name, image.copy(), imobj)


def build_transforms(pixel_mean, pixel_std):
    to_bgr = True

    normalize_transform = Normalize(
        mean=pixel_mean, std=pixel_std, to_bgr=to_bgr
    )

    transform = Compose(
        [
            ToTensor(),
            normalize_transform,
        ]
    )
    return transform


def build_transforms2(pixel_mean, pixel_std):
    to_bgr = True

    normalize_transform = Normalize(
        mean=pixel_mean, std=pixel_std, to_bgr=to_bgr
    )

    transform = Compose(
        [
            ConvertToFloat(),
            PhotometricDistort(0.5),
            ToTensor(),
            normalize_transform
            
        ]
    )
    return transform