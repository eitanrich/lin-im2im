import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def mosaic(X, name=None, show=False, clip=True, to_uint8=True, ch_first=False, n_cols=None):
    is_flat = len(X.shape) == 2
    w = int(np.sqrt(X.shape[1]/3)) if is_flat else X.shape[2]
    if ch_first:
        if is_flat:
            X = X.reshape(-1, 3, w, w)
        X = np.transpose(X, [0, 2, 3, 1])
    elif is_flat:
        X = X.reshape(-1, w, w, 3)

    if clip:
        X = np.clip(X, 0.0, 1.0)
    n_cols = n_cols or int(np.sqrt(X.shape[0]))
    n_rows = X.shape[0] // n_cols
    mosaic_image = np.vstack([np.hstack([X[i*n_cols+j] for j in range(n_cols)]) for i in range(n_rows)])
    if show:
        plt.figure()
        plt.imshow(mosaic_image)
        plt.axis('off')
        if name:
            plt.title(name)
    if to_uint8:
        mosaic_image = (mosaic_image * 255.0).astype(np.uint8)
    return mosaic_image
