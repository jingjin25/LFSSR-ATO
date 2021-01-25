
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
from os import path
from os.path import join
from collections import defaultdict
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import dataset, util
from model import model_LFSSR
#--------------------------------------------------------------------------#
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)
        
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Light Field SR")

#training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--max_epoch", type=int, default=300, help="maximum epoch for training")
parser.add_argument("--num_cp", type=int, default=5, help="Number of epoches for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epoches for saving loss figure")
parser.add_argument("--dataset", type=str, default="", help="Dataset for training")
parser.add_argument("--dataset_path", type=str, default="LFData/train_all.h5", help="Dataset file for training")
parser.add_argument("--angular_num", type=int, default=9, help="Size of angular dim")
parser.add_argument("--scale", type=int, default=4, help="SR factor")
parser.add_argument("--feature_num", type=int, default=64, help="number of feature channels")
parser.add_argument('--layer_num', action=Store_as_array, type=int, nargs='+',help="number of layers in resBlocks")
parser.add_argument('--layer_num_refine', type=int, default=3, help="number of refine SAS layers")
parser.add_argument('--weight_epi', type=float, default=0.1, help="weight for epi loss")
parser.add_argument('--ATO_path', type=str, default="pretrained_models/ATONet_2x.pth", help="pretrained ATO model path")

opt = parser.parse_args()


def main():
    print(opt)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1)

    # Data loader
    print('===> Loading datasets')
    train_set = dataset.TrainDataFromHdf5_LF(opt.dataset_path, opt.scale, opt.patch_size, opt.angular_num)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))

    # Build model
    print("building net")
    model = model_LFSSR.LFSSRNet(opt).to(device)

    # pretrained ATONet
    pt_srnet = torch.load(opt.ATO_path)
    pt_dict = pt_srnet['model']
    # model_dict = model.state_dict()
    # state_dict = {k: v for k, v in pt_dict.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    model.load_state_dict(pt_dict, strict=False)

    for n,p in model.named_parameters():
        if n in pt_dict:
            p.requires_grad = False

    # optimizer and loss logger
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)

    # model dir
    # model_dir = 'model_x{}_{}{}x{}_fn{}_ln{}_refine{}_epi{}_lr{}_step{}x{}'.format(opt.scale,opt.dataset,opt.angular_num,opt.angular_num,opt.feature_num,opt.layer_num,opt.layer_num_refine,opt.weight_epi,opt.lr,opt.step,opt.reduce)
    model_dir = 'checkpoint_LFSSR_{}x_w{}'.format(opt.scale, opt.weight_epi)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # optionally resume from a checkpoint
    if opt.resume_epoch:
        resume_path = join(model_dir,'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            losslogger = checkpoint['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    # training
    print("===> training")
    for epoch in range(opt.resume_epoch + 1, opt.max_epoch):
        model.train()
        scheduler.step()
        loss_count = 0.

        for k in range(50):
            for i, batch in enumerate(train_loader, 1):

                lf_hr = batch[0].to(device) #[N,an2,h,w]
                lf_lr = batch[1].to(device) #[N,an2,h//s,w//s]

                lf_out = model(lf_lr)

                loss = L1_Charbonnier_loss(lf_out, lf_hr) + opt.weight_epi * epi_loss(lf_out, lf_hr)

                loss_count += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losslogger['epoch'].append(epoch)
        losslogger['loss'].append(loss_count/len(train_loader))

        # checkpoint
        if epoch % opt.num_cp == 0:
            model_save_path = join(model_dir, "model_epoch_{}.pth".format(epoch))
            state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'losslogger': losslogger,}
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

        if epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'],losslogger['loss'])
            plt.savefig(model_dir+".jpg")
            plt.close()


def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.sum(error) / torch.numel(error)
    return loss


def epi_loss(pred, label):

    def gradient(pred):
        D_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        D_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        D_day = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
        D_dax = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        return D_dx, D_dy, D_dax, D_day


    N, an2, h, w = pred.shape
    an = int(math.sqrt(an2))
    pred = pred.view(N, an, an, h, w)
    label = label.view(N, an, an, h, w)

    pred_dx, pred_dy, pred_dax, pred_day = gradient(pred)
    label_dx, label_dy, label_dax, label_day = gradient(label)

    return L1_Charbonnier_loss(pred_dx, label_dx) + L1_Charbonnier_loss(pred_dy, label_dy) + L1_Charbonnier_loss(pred_dax,label_dax) + L1_Charbonnier_loss(pred_day, label_day)


if __name__ == "__main__":
    main()
