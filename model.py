# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from layer import *
import math
import torchvision

class ResNet(nn.Module):
  def __init__(self, in_channels, out_channels, norm="bnorm"):
    super(ResNet, self).__init__()

    self.learning_type = learning_type

    self.layer1 = CBR2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True, norm=None, relu=0.0)
    self.pool = nn.MaxPool2d(3,2,1)

    self.layer2 = nn.Sequential(
        ResBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False)
    )

    self.layer3 = nn.Sequential(
        ResBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=True),
        ResBlock(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False)
    )
    self.layer4 = nn.Sequential(
        ResBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=True),
        ResBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False)
    )
    self.layer5 = nn.Sequential(
        ResBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=True),
        ResBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False),
        ResBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True, norm="bnorm", relu=0.0, downsample=False)
    )

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512 * out_channels, 1)



  def forward(self, x):
    
    x = self.layer1(x)
    x = self.pool(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.avgpool(x)
    out = self.fc(x)

    return out


# MobileNet_v2
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=2, input_size=112, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True, input_size=112):
    
    model = MobileNetV2(width_mult=1, input_size=112, n_class=2)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


class AdditionalModule(nn.Module):
    def __init__(self, n_classes=2):
        self.backbone = torchvision.models.resnet50(pretrained=True)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.layer1 = nn.Linear(2048, 256) # outchannel=256, relu
        self.relu = nn.ReLU()
        self.dropout = Dropout(0.5)
        self.layer2 = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.backbone.output(x)


