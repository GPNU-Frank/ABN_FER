
import torch
from torch import nn
import torchvision
import sys
sys.path.append('..')
from models.st_gcn import Model
import torch.nn.functional as F

class ResnetOri(nn.Module):
    def __init__(self, num_classes):
        super(ResnetOri, self).__init__()

        model = torchvision.models.resnet18(False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.fc = nn.Linear(512, 6)

        model._modules.get('layer4').register_forward_hook(self.get_layer4_feature)
        self.resnet = model

    def get_layer4_feature(self, module, input, output):
        self.layer4_feature = output

    def forward(self, x):
        out = self.resnet(x)
        return out, self.layer4_feature

    def reset_all_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class RensnetAndSTGCNCAM(nn.Module):
    def __init__(self, num_classes):
        super(RensnetAndSTGCNCAM, self).__init__()

        self.resnet = ResnetOri(num_classes)
        self.st_gcn = Model(2, 6, {}, False)
        self.fc = nn.Linear(512, num_classes)
        self.res_softmax_weight = list(self.resnet.resnet.parameters())[-2]


    def forward(self, inputs):
        img, landmark = inputs
        feat_img, layer4_feature = self.resnet(img)
        feat_lm = self.st_gcn(landmark)

        resnet_predict = F.softmax(feat_img, dim=0).argmax(dim=1)
        self.get_layer4_xy(layer4_feature, feat_img, landmark, resnet_predict)
        x = torch.cat([feat_img, feat_lm], 1)
        out = self.fc(x)
        return out

    def get_layer4_xy(self, layer4_feature, out_img, landmark, resnet_predict):

        bz, nc, h, w = layer4_feature.shape
        # layer4_feature = layer4_feature.reshape(())
        cam = [None] * bz
        for i in range(bz):
            cam[i] = torch.einsum('ijkl, j -> ikl', layer4_feature, self.res_softmax_weight[resnet_predict[i]])
            # cam[i] = self.res_softmax_weight[resnet_predict[i]] * layer4_feature
            cam[i] = cam[i].reshape(bz, 128, 128)
        cam = torch.stack(cam, dim=0)
        print(cam.shape)

        

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
    lm = torch.rand((32, 2, 1, 55), dtype=torch.float)
    model = RensnetAndSTGCNCAM(num_classes=6)
    out = model((img, lm))
    print(model)
    print(out.size())