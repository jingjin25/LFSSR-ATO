
import torch
import numpy as np
import math
import argparse
import copy


class StoreAsArray(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


def ycbcr2rgb(ycbcr):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:, 0] -= 16. / 255.
    rgb[:, 1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)


def CropPatches(image, len, crop):
    # left [1,an2,h,lw]
    # middles[n,an2,h,mw]
    # right [1,an2,h,rw]
    an, h, w = image.shape[1:4]
    left = image[:, :, :, 0:len + crop]
    num = math.floor((w - len - crop) / len)
    middles = torch.Tensor(num, an, h, len + crop * 2).to(image.device)
    for i in range(num):
        middles[i] = image[0, :, :, (i + 1) * len - crop:(i + 2) * len + crop]
    right = image[:, :, :, -(len + crop):]
    return left, middles, right


def MergePatches(left, middles, right, h, w, len, crop):
    n, a = left.shape[0:2]
    # out = torch.Tensor(n, a, h, w).to(left.device)
    out = np.zeros((n,a,h,w)).astype(left.dtype)
    out[:, :, :, :len] = left[:, :, :, :-crop]
    for i in range(middles.shape[0]):
        out[:, :, :, len * (i + 1):len * (i + 2)] = middles[i:i + 1, :, :, crop:-crop]
    out[:, :, :, -len:] = right[:, :, :, crop:]
    return out


def compt_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

