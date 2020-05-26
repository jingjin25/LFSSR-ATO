import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os.path import join

import math
import copy
import pandas as pd
import time 

import h5py
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from skimage.measure import compare_ssim  

from model.model_LFSSR import LFSSRNet
from utils import dataset, util


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test settings
parser = argparse.ArgumentParser(description="LFSSR-ATO demo")
parser.add_argument("--model_dir", type=str, default="pretrained_models", help="folder containing the pretrained models")
parser.add_argument("--save_dir", type=str, default="results", help="folder to save the test results")
parser.add_argument("--scale", type=int, default=4, help="SR factor")
parser.add_argument("--test_dataset", type=str, default="", help="dataset for test")
parser.add_argument("--angular_num", type=int, default=9, help="Size of angular dim")
parser.add_argument("--save_img", type=int, default=0, help="save image or not")
parser.add_argument("--crop", type=int, default=0, help="crop the image into patches when out of memory")

parser.add_argument("--feature_num", type=int, default=64, help="number of feature channels")
parser.add_argument('--layer_num', action=util.StoreAsArray, type=int, nargs='+', help="number of layers in resBlocks")
parser.add_argument('--layer_num_refine', type=int, default=3, help="number of refine SAS layers")

opt = parser.parse_args()
print(opt)


def main():
    # generate save folder
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    if opt.save_img:
        opt.save_img_dir = '{}/saveImg_LFSSR/{}_x{}'.format(opt.save_dir, opt.test_dataset, opt.scale)
        if not os.path.exists(opt.save_img_dir):
            os.makedirs(opt.save_img_dir)
    opt.csv_name = '{}/res_LFSSR_{}_x{}.csv'.format(opt.save_dir, opt.test_dataset, opt.scale)

    # Data loader
    print('===> Loading test datasets')
    data_path = join('LFData', 'test_{}_x{}.h5'.format(opt.test_dataset, opt.scale))
    test_set = dataset.TestDataFromHdf5(data_path, opt.scale)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('loaded {} LFIs from {}'.format(len(test_loader), data_path))

    # Build model
    print("===> Building net")
    model = LFSSRNet(opt).to(device)

    # load state dict
    resume_path = join(opt.model_dir, "LFSSRNet_{}x.pth".format(opt.scale))
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    print('loaded pretrained model'.format(resume_path))

    # testing
    print("===> testing")
    model.eval()

    lf_list = []
    lf_psnr_y_list = []
    lf_ssim_y_list = []

    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            # SR
            gt_y, sr_ycbcr, lr_y = batch[0].numpy(), batch[1].numpy(), batch[2]

            lr_y = lr_y.to(device)

            start = time.time()

            if not opt.crop:
                sr_y = model(lr_y)
                sr_y = sr_y.cpu().numpy()
            else:
                crop = 8
                length = 120
                lr_l, lr_m, lr_r = util.CropPatches(lr_y, length//opt.scale, crop//opt.scale)
                sr_l = model(lr_l).cpu().numpy()
                sr_m = np.zeros((lr_m.shape[0], opt.angular_num*opt.angular_num, lr_m.shape[2]*opt.scale, lr_m.shape[3]*opt.scale), dtype=np.float32)
                for i in range(lr_m.shape[0]):
                    sr_m[i:i+1] = model(lr_m[i:i+1]).cpu().numpy()
                sr_r = model(lr_r).cpu().numpy()
                sr_y = util.MergePatches(sr_l, sr_m, sr_r, lr_y.shape[2]*opt.scale, lr_y.shape[3]*opt.scale, length, crop)

            end = time.time()
            print('running time: ', end - start)

            # save results
            lf_psnr, lf_ssim = save_results(sr_y, sr_ycbcr, gt_y, k)

            lf_list.append(k)
            lf_psnr_y_list.append(lf_psnr)
            lf_ssim_y_list.append(lf_ssim)

    dataframe_lfi = pd.DataFrame({'lfiNo': lf_list, 'psnr Y':lf_psnr_y_list, 'ssim Y':lf_ssim_y_list})
    dataframe_lfi.to_csv(opt.csv_name, index=False, sep=',', mode='a')
    dataframe_lfi = pd.DataFrame({'summary': ['avg'], 'psnr Y':[np.mean(lf_psnr_y_list)], 'ssim Y':[np.mean(lf_ssim_y_list)]})
    dataframe_lfi.to_csv(opt.csv_name, index=False, sep=',', mode='a')


def save_results(sr_y, sr_ycbcr, gt_y, lf_no):

    sr_ycbcr[:, :, 0] = sr_y

    view_list = []
    view_psnr_y_list = []
    view_ssim_y_list = []

    for i in range(opt.angular_num * opt.angular_num):
        if opt.save_img:
            save_img_dir = '{}/saveImg_ATO/{}_x{}'.format(opt.save_dir, opt.test_dataset, opt.scale)
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)

            img_name = '{}/SR{}_view{}.png'.format(opt.save_img_dir, lf_no, i)
            sr_rgb_temp = util.ycbcr2rgb(np.transpose(sr_ycbcr[0, i], (1, 2, 0)))
            img = (sr_rgb_temp.clip(0, 1) * 255.0).astype(np.uint8)
            Image.fromarray(img).convert('RGB').save(img_name)

        cur_psnr = util.compt_psnr(gt_y[0, i], sr_y[0, i])
        cur_ssim = compare_ssim((gt_y[0, i] * 255.0).astype(np.uint8), (sr_y[0, i] * 255.0).astype(np.uint8),
                                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

        view_list.append(i)
        view_psnr_y_list.append(cur_psnr)
        view_ssim_y_list.append(cur_ssim)

    dataframe_lfi = pd.DataFrame({'View_LFI{}'.format(lf_no): view_list, 'psnr Y': view_psnr_y_list, 'ssim Y': view_ssim_y_list})
    dataframe_lfi.to_csv(opt.csv_name, index=False, sep=',', mode='a')
    return np.mean(view_psnr_y_list), np.mean(view_ssim_y_list)




if __name__ == '__main__':

    main()