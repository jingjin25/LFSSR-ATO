import torch
import torch.nn as nn
import torch.nn.init as init

class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


def make_layer(block, nf, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf))
    return nn.Sequential(*layers)


class AltFilter(nn.Module):
    def __init__(self, an):
        super(AltFilter, self).__init__()

        self.an = an
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        N, c, h, w = x.shape  # [N*an2,c,h,w]
        N = N // (self.an * self.an)

        out = self.relu(self.spaconv(x))  # [N*an2,c,h,w]
        out = out.view(N, self.an * self.an, c, h * w)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * h * w, c, self.an, self.an)  # [N*h*w,c,an,an]

        out = self.relu(self.angconv(out))  # [N*h*w,c,an,an]
        out = out.view(N, h * w, c, self.an * self.an)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * self.an * self.an, c, h, w)  # [N*an2,c,h,w]

        return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()

