
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
        model.fc = nn.Linear(512, 256)
        self.resnet = model

    def forward(self, x):
        out = self.resnet(x)
        return out


class RensnetAndSTGCN(nn.Module):
    def __init__(self, num_classes):
        super(RensnetAndSTGCN, self).__init__()

        self.resnet = ResnetOri(num_classes)
        self.st_gcn = Model(2, 6, {}, False)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, inputs):
        img, landmark = inputs
        feat_img = self.resnet(img)
        feat_lm = self.st_gcn(landmark)
        x = torch.cat([feat_img, feat_lm], 1)
        out = self.fc(x)
        return out

    def reset_all_weights(self):
        def reset_layer_weights(layer):
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.apply(reset_layer_weights)


    # def reset_all_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    


if __name__ == '__main__':
    img = torch.rand((32, 1, 128, 128), dtype=torch.float)
    lm = torch.rand((32, 2, 1, 51), dtype=torch.float)
    model = RensnetAndSTGCN(num_classes=6)
    out = model((img, lm))
    print(model)
    print(out.size())