# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn


class CBR2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()
        
        layers = []

        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)]
        if not norm is None:
            if norm == "bnorm":
              layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
              layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None:
            layers += [nn.ReLU(inplace=True) if relu == 0.0 else nn.LeakyRelu(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.downsample = downsample

        if self.downsample:
            layers = []
            # 1st CBR2d
            layers += [CBR2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias, norm=norm, relu=relu)]
            # 2nd CBR2d
            layers += [CBR2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias, norm=norm, relu=None)]
            self.downsize = nn.Conv2d(in_channels, out_channels, stride=2, padding=0)
        else:
            layers = []
            # 1st CBR2d
            layers += [CBR2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias, norm=norm, relu=relu)]
            # 2nd CBR2d
            layers += [CBR2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias, norm=norm, relu=None)]
            self.make_equal_channel = nn.Conv2d(in_channels, out_channels, stride=1, padding=0)

        self.resblk = nn.Sequential(*layers)


    def forward(self, x):
        if self.downsample:
            out = self.resblk(x)
            x = self.downsize(out)
            x = nn.BatchNorm2d(num_features=out_channels)
            return nn.ReLU(out + x, inplace=True)
        else:
            out = self.resblk(x)
            if x.size() is not out.size():
                x = self.make_equal_channel(out)
            x = nn.BatchNorm2d(num_features=out_channels)
            return nn.ReLU(out + x, inplace=True)