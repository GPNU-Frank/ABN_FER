# from __future__ import absolute_import


import torch
import torch.nn as nn
import math
from torch.nn import Module
from torch.nn.parameter import Parameter

from .utils.graph import Graph
from .utils.tgcn import ConvTemporalGraphical

import torch.nn.functional as F

# __all__ = ['resnet']

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


class ResNetAndResGCN(nn.Module):

    def __init__(self, depth, num_classes=1000, gcn_hidden=512, gcn_out=64, dropout=0.5):
        super(ResNetAndResGCN, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        graph = Graph({})
        self.A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.A = self.A.cuda()

        # self.graph = Graph({})

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, n, down_size=True)

        self.gcn11 = ConvTemporalGraphical(64, 256, 1)
        self.gcn12 = ConvTemporalGraphical(256, 64, 1)

        self.layer2 = self._make_layer(block, 128, n, stride=2, down_size=True)
        # self.upsample2 = nn.functional.upsample()
        self.gcn21 = ConvTemporalGraphical(128, 256, 1)
        self.gcn22 = ConvTemporalGraphical(256, 128, 1)
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
        self.sigmoid = nn.Sigmoid()

        self.layer3 = self._make_layer(block, 256, n, stride=2, down_size=True)

        # self.gcn31 = ConvTemporalGraphical(256, 512, 1)
        # self.gcn32 = ConvTemporalGraphical(512, 256, 1)
        # self.avgpool = nn.AvgPool2d(8)

        self.layer4 = self._make_layer(block, 256, n, stride=2, down_size=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.fc = nn.Linear(256 + 128 + 64, num_classes)

        # self.fc1 = nn.Linear(256, num_classes)

        # print(block, n)
        # self.att_layer3 = self._make_layer(block, 64, n, stride=1)
        # self.att_conv   = nn.Conv2d(64, num_classes, kernel_size=3, padding=1,
        #                        bias=False)
        # self.att_gap = nn.AvgPool2d(16)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

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


    def get_landmark_feature(self, feature_map, landmark, ratio=1, img_size=(128, 128)):
        # feature_map: bz, c, h, w
        # landmark: bz, (x, y), 55
        # 注意 grad
        rows, cols = img_size
        rows //= ratio
        cols //= ratio
        with torch.no_grad():

            bz, c, h, w = feature_map.size()
            feature_map = feature_map.view(bz, c, -1)

            lm_index = []
            bz, _, lm_num = landmark.size()

            landmark = landmark // ratio
            for i in range(bz):
                lm_index.append([])
                for j in range(lm_num):
                    lm_index[i].append(landmark[i, 1, j] * cols + landmark[i, 0, j])

            lm_index = torch.Tensor(lm_index)
            # lm_index = lm_index.as
            lm_index = lm_index.unsqueeze(1)
            lm_index = lm_index.expand(-1, c, -1).long()
            # lm_index = lm_index.numpy()
            # landmark = landmark // ratio
            # channel_size = feature_map.size()[1]
            # landmark = landmark.unsqueeze(1)
            # landmark = landmark.expand(-1, 32, -1, -1)
            # landmark_x = landmark[:, :, 0, :]
            # landmark_y = landmark[:, :, 1, :] 
            # bz, c, h, w = feature_map.size()
            # feature_map = feature_map.view(bz, c, -1)
            
            # 把 lm_index 放入 cuda
            lm_index = lm_index.cuda()

            nodes = torch.gather(feature_map, dim=2, index=lm_index)
            # print(nodes.shape)
            # print(nodes)
            
            # ()reshape 成 (lm_nums, feat)
            # nodes = nodes.permute()
            nodes = nodes.unsqueeze(2)
            return nodes

    def forward(self, x, landmark):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 128 * 128

        x = self.layer1(x)  # 128 * 128
        bz, c, h, w = x.size()
        # x = F.upsample(x, (128, 128))
        node1 = self.get_landmark_feature(x, landmark, ratio=2)
        x_gcn1, _ = self.gcn11(node1, self.A)
        x_gcn1, _ = self.gcn12(x_gcn1, self.A)
        # residual
        x_gcn1 += node1

        x_gcn1 = F.adaptive_avg_pool2d(x_gcn1, 1)
        x_gcn1 = x_gcn1.squeeze(-1).squeeze(-1)

        x = self.layer2(x)  # 64 * 64
        bz, c, h, w = x.size()
        # x = F.upsample(x, (128, 128))
        node2 = self.get_landmark_feature(x, landmark, ratio=4)
        x_gcn2, _ = self.gcn21(node2, self.A)
        x_gcn2, _ = self.gcn22(x_gcn2, self.A)
        # residual
        x_gcn2 += node2

        x_gcn2 = F.adaptive_avg_pool2d(x_gcn2, 1)
        x_gcn2 = x_gcn2.squeeze(-1).squeeze(-1)

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
        rx = self.layer3(rx)  # 32 * 32
        rx = self.layer4(rx)
        # bz, c, h, w = rx.size()
        # # rx = F.upsample(rx, (128, 128))
        # node3 = self.get_landmark_feature(rx, landmark, ratio=8)
        # x_gcn3, _ = self.gcn31(node3, self.A)
        # x_gcn3, _ = self.gcn32(x_gcn3, self.A)
        # # residual
        # x_gcn3 += node3

        # x_gcn3 = F.adaptive_avg_pool2d(x_gcn3, 1)
        # x_gcn3 = x_gcn3.squeeze(-1).squeeze(-1)

        # rx = self.layer4(rx)
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)

        rx = torch.cat([rx, x_gcn1, x_gcn2], dim=1)
        rx = self.fc(rx)
        # rx = F.relu(rx)
        # rx = F.dropout(rx, 0.7)
        # rx = self.fc1(rx)

        return rx

    def reset_all_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# def resnet(**kwargs):
#     """
#     Constructs a ResNet model.
#     """
#     return ResNet(**kwargs)


if __name__ == '__main__':
    inputs = torch.rand((32, 1, 128, 128))
    landmarks = torch.randint(0, 128, (32, 2, 55))

    model = ResNetAndGCN(20, num_classes=6)

    # feature_map = torch.rand((32, 32, 128, 128))

    # feature_map = torch.ones((2, 2, 2, 2))
    # feature_map[:, :, 0, 1] += 1
    # feature_map[:, :, 1, 0] += 2
    # feature_map[:, :, 1, 1] += 3
    # # feature_map[:, :, 0, 1] += 1

    # print(feature_map)

    # landmarks = torch.ones((2, 2, 3))
    # landmarks[:, 0, 0] = 1
    # landmarks[:, 1, 0] = 1

    # landmarks[:, 0, 1] = 0
    # landmarks[:, 1, 1] = 1
    
    # landmarks[:, 0, 2] = 1
    # landmarks[:, 1, 2] = 0

    # # landmarks[:, 0, 3] = 0
    # # landmarks[:, 1, 3] = 0

    # model.get_landmark_feature(feature_map, landmarks, img_size=(2, 2))

    outs = model(inputs, landmarks)
    print(outs)