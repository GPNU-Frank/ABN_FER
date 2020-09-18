
import torch
from torch import nn
import torchvision
from .st_gcn import Model


class ResnetOri(nn.Module):
    def __init__(self, num_classes):
        super(ResnetOri, self).__init__()

        model = torchvision.models.resnet18(False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, 6)
        self.resnet = model

    def forward(self, x):
        out = self.resnet(x)
        return out

    def reset_all_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
