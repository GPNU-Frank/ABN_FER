
import torch
from torch import nn
import torchvision


class ResnetOri(nn.Module):
    def __init__(self, num_classes):
        super(ResnetOri, self).__init__()

        model = torchvision.models.resnet18(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, num_classes)
        self.resnet = model

    def forward(self, x):
        out = self.resnet(x)
        return out


def resnet_ori(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResnetOri(**kwargs)


# if __name__ == '__main__':
#     model = resnet_ori(num_classes=6)
#     print(model)