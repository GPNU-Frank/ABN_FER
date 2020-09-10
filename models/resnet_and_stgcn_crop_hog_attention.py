
import torch
from torch import nn
import torchvision
from .st_gcn import Model
import math

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



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResnetHog(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResnetHog, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, down_size=True)
        self.layer2 = self._make_layer(block, 32, n, stride=2, down_size=True)

        # self.att_layer3 = self._make_layer(block, 64, n, stride=1, down_size=False)
        # self.bn_att = nn.BatchNorm2d(64 * block.expansion)
        # self.att_conv   = nn.Conv2d(64 * block.expansion, num_classes, kernel_size=1, padding=0,
        #                        bias=False)
        # self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
        #                        bias=False)
        # self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
        #                        bias=False)
        # self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        # self.sigmoid = nn.Sigmoid()

        self.layer3 = self._make_layer(block, 64, n, stride=2, down_size=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # print(block, n)
        # self.att_layer3 = self._make_layer(block, 64, n, stride=1)
        # self.att_conv   = nn.Conv2d(64, num_classes, kernel_size=3, padding=1,
        #                        bias=False)
        # self.att_gap = nn.AvgPool2d(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16

        # ax = self.bn_att(self.att_layer3(x))
        # ax = self.relu(self.bn_att2(self.att_conv(ax)))
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        # # self.att = self.att.view(bs, 1, ys, xs)
        # ax = self.att_conv2(ax)
        # ax = self.att_gap(ax)
        # ax = ax.view(ax.size(0), -1)

        # rx = x * self.att
        # rx = rx + x
        rx = x
        rx = self.layer3(rx)  # 8x8
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        # rx = self.fc(rx)

        return rx


class ResnetCrop(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResnetCrop, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, down_size=True)
        self.layer2 = self._make_layer(block, 32, n, stride=2, down_size=True)

        # self.att_layer3 = self._make_layer(block, 64, n, stride=1, down_size=False)
        # self.bn_att = nn.BatchNorm2d(64 * block.expansion)
        # self.att_conv   = nn.Conv2d(64 * block.expansion, num_classes, kernel_size=1, padding=0,
        #                        bias=False)
        # self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
        #                        bias=False)
        # self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
        #                        bias=False)
        # self.bn_att3 = nn.BatchNorm2d(1)
        # self.att_gap = nn.AvgPool2d(16)
        # self.sigmoid = nn.Sigmoid()

        self.layer3 = self._make_layer(block, 64, n, stride=2, down_size=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # print(block, n)
        # self.att_layer3 = self._make_layer(block, 64, n, stride=1)
        # self.att_conv   = nn.Conv2d(64, num_classes, kernel_size=3, padding=1,
        #                        bias=False)
        # self.att_gap = nn.AvgPool2d(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16

        # ax = self.bn_att(self.att_layer3(x))
        # ax = self.relu(self.bn_att2(self.att_conv(ax)))
        # bs, cs, ys, xs = ax.shape
        # self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        # # self.att = self.att.view(bs, 1, ys, xs)
        # ax = self.att_conv2(ax)
        # ax = self.att_gap(ax)
        # ax = ax.view(ax.size(0), -1)

        # rx = x * self.att
        # rx = rx + x
        rx = x
        rx = self.layer3(rx)  # 8x8
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        # rx = self.fc(rx)

        return rx


class RensnetAndSTGCNAndCropHOGAttention(nn.Module):
    def __init__(self, num_classes):
        super(RensnetAndSTGCNAndCropHOGAttention, self).__init__()

        self.resnet_hog = ResnetHog(20, num_classes)
        # self.resnet = ResnetOri(num_classes)
        self.resnet_crop = ResnetCrop(20, num_classes)
        self.st_gcn = Model(2, 6, {}, False)
        self.fc = nn.Linear(192, num_classes)
        self.att_fc = nn.Linear(64, 1)

    def forward(self, inputs):
        img, landmark, crop = inputs
        feat_img = self.resnet_hog(img)
        feat_crop = self.resnet_crop(crop)
        feat_lm = self.st_gcn(landmark)
        
        att_img = self.att_fc(feat_img)
        att_lm = self.att_fc(feat_lm)
        att_crop = self.att_fc(feat_crop)



        x = torch.cat([feat_img * att_img, feat_lm * att_lm, feat_crop * att_crop], 1)
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
    crop = torch.rand((32, 4, 24, 32), dtype=torch.float)
    lm = torch.rand((32, 2, 1, 51), dtype=torch.float)
    model = RensnetAndSTGCNAndCrop(num_classes=6)
    out = model((img, lm, crop))
    print(model)
    print(out.size())