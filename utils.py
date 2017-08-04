"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import random
import scipy.misc
import numpy as np
import copy

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []
    def __call__(self, image):
        if self.maxsize == 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img=self.num_img+1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp = copy.copy(self.images[idx])
            self.images[idx] = image
            return tmp
        else:
            return image

# -----------------------------
