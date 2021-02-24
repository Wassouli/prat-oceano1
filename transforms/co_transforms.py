import numbers
import random
import numpy as np
# from scipy.misc import imresize
from skimage.transform import resize as imresize
import scipy.ndimage as ndimage


def get_co_transforms(aug_args):
    transforms = []
    if aug_args.crop:
        transforms.append(RandomCrop(aug_args.para_crop))
    if aug_args.hflip:
        transforms.append(RandomHorizontalFlip())
    if aug_args.swap:
        transforms.append(RandomSwap())
    return Compose(transforms)


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input):
        for t in self.co_transforms:
            input = t(input)
        return input

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = [img[y1: y1 + th, x1: x1 + tw] for img in inputs]
       
        return inputs


class RandomSwap(object):
    def __call__(self, inputs):
        n = len(inputs)
        if random.random() < 0.5:
            inputs = inputs[::-1]
          
        return inputs


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs):
        if random.random() < 0.5:
            inputs = [np.copy(np.fliplr(im)) for im in inputs]
            
        return inputs