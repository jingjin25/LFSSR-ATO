
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
from os.path import join
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import dataset, util
from model import model_ATO

# Training settings
parser = argparse.ArgumentParser(description="LFSSR All-to-One")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--max_epoch", type=int, default=700, help="maximum epoch for training")
parser.add_argument("--num_cp", type=int, default=20, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")
parser.add_argument("--dataset", type=str, default="all", help="Dataset for training")
parser.add_argument("--dataset_path", type=str, default="LFData/train_all.h5", help="Dataset file for training")
parser.add_argument("--angular_num", type=int, default=7, help="Size of angular dim")
parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--feature_num", type=int, default=64, help="number of feature channels")
parser.add_argument('--layer_num', action=util.StoreAsArray, type=int, nargs='+', help="number of layers in resBlocks")

opt = parser.parse_args()
# print(opt)


def main():
    print(opt)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1)

    # Data loader
    print('===> Loading datasets')
    train_set = dataset.TrainDataFromHdf5(opt.dataset_path, opt.scale, opt.patch_size, opt.angular_num)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))

    # Build model
    print("===> building net")
    model = model_ATO.ATONet(opt).to(device)

    # optimizer and loss logger
    print("===> setting optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)

    # model dir
    model_dir = 'checkpoint_ATO_{}x'.format(opt.scale)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # optionally resume from a checkpoint
    if opt.resume_epoch:
        resume_path = join(model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
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
                view_hr = batch[0].to(device)  # [N,1,h,w]
                lf_lr = batch[1].to(device)  # [N,an2,h//s,w//s]
                ref_ind = batch[-1].to(device)  # [N]

                view_sr = model(lf_lr, ref_ind)

                loss = L1_Charbonnier_loss(view_sr, view_hr)
                loss_count += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losslogger['epoch'].append(epoch)
        losslogger['loss'].append(loss_count / len(train_loader))


        # checkpoint
        if epoch % opt.num_cp == 0:
            model_save_path = join(model_dir, "model_epoch_{}.pth".format(epoch))
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger, }
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

        # loss snapshot
        if epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(model_dir + ".jpg")
            plt.close()


def L1_Charbonnier_loss(X, Y):
        eps = 1e-6
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + eps)
        loss = torch.sum(error) / torch.numel(error)
        return loss


if __name__ == "__main__":
    main()
 




