# from __future__ import absolute_import


import torch
import torch.nn as nn
import math
from torch.nn import Module
from torch.nn.parameter import Parameter

# from .utils.graph import Graph
# from .utils.tgcn import ConvTemporalGraphical

import torch.nn.functional as F
from collections import OrderedDict


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


class ResNetAndDGCNN(nn.Module):

    def __init__(self, depth, num_classes=1000, gcn_hidden=512, gcn_out=64, dropout=0.5):
        super(ResNetAndDGCNN, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock



        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, n, down_size=True)

        self.edge_conv_1 = EdgeConv(layers=[2, 256, 256, 64])
        self.edge_conv_2 = EdgeConv(layers=[64, 128])
        self.conv_block = conv_bn_block(128, 1024, 1)
        self.fc_block = fc_bn_block(1024, 512)
        self.fc = nn.Linear(512, num_classes)

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
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.layer1(x)

        # node = self.get_landmark_feature(x, landmark, ratio=2)
        # node = node.squeeze(2)
        landmark = landmark.permute(0, 2, 1)
        B, N, C = landmark.shape

        x = self.edge_conv_1(landmark)
        x = self.edge_conv_2(x)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = nn.AvgPool1d(N)(x)
        x = x.squeeze(2)
        x = self.fc_block(x)
        x = self.fc(x)
        return x



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


def conv_bn_block(input, output, kernel_size):
    '''
    标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    '''
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


def fc_bn_block(input, output):
    '''
    标准全连接块（fc + bn + relu）
    :param input:  输入通道数
    :param output:  输出通道数
    :return:  卷积核大小
    '''
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


class EdgeConv(nn.Module):
    '''
    EdgeConv模块
    1. 输入为：n * f
    2. 创建KNN graph，变为： n * k * f
    3. 接上若干个mlp层：a1, a2, ..., an
    4. 最终输出为：n * k * an
    5. 全局池化，变为： n * an
    '''
    def __init__(self, layers, K=3):
        '''
        构造函数
        :param layers: e.p. [3, 64, 64, 64]
        :param K:
        '''
        super(EdgeConv, self).__init__()

        self.K = K
        self.layers = layers
        # self.KNN_Graph = torch.zeros(Args.batch_size, 2048, self.K, self.layers[0]).to(Args.device)

        if layers is None:
            self.mlp = None
        else:
            mlp_layers = OrderedDict()
            for i in range(len(self.layers) - 1):
                if i == 0:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(2*self.layers[i], self.layers[i + 1], 1)
                else:
                    mlp_layers['conv_bn_block_{}'.format(i+1)] = conv_bn_block(self.layers[i], self.layers[i+1], 1)
            self.mlp = nn.Sequential(mlp_layers)

    def createSingleKNNGraph(self, X):
        '''
        generate a KNN graph for a single point cloud
        :param X:  X is a Tensor, shape: [N, F]
        :return: KNN graph, shape: [N, K, F]
        '''
        N, F = X.shape
        assert F == self.layers[0]

        # self.KNN_Graph = np.zeros(N, self.K)

        # 计算距离矩阵
        dist_mat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N) + \
                   torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N).t()
        dist_mat.addmm_(1, -2, X, X.t())

        # 对距离矩阵排序
        dist_mat_sorted, sorted_indices = torch.sort(dist_mat, dim=1)
        # print(dist_mat_sorted)

        # 取出前K个（除去本身）
        knn_indexes = sorted_indices[:, 1:self.K+1]
        # print(sorted_indices)

        # 创建KNN图
        knn_graph = X[knn_indexes]

        return knn_graph

    def forward(self, X):
        '''
        前向传播函数
        :param X:  shape: [B, N, F]
        :return:  shape: [B, N, an]
        '''
        # print(X.shape)
        B, N, F = X.shape
        assert F == self.layers[0]

        KNN_Graph = torch.zeros(B, N, self.K, self.layers[0]).cuda()
        # KNN_Graph = torch.zeros(B, N, self.K, self.layers[0])

        # creating knn graph
        # X: [B, N, F]
        for idx, x in enumerate(X):
            # x: [N, F]
            # knn_graph: [N, K, F]
            # self.KNN_Graph[idx] = self.createSingleKNNGraph(x)
            KNN_Graph[idx] = self.createSingleKNNGraph(x)
        # print(self.KNN_Graph.shape)
        # print('KNN_Graph: {}'.format(KNN_Graph[0][0]))

        # X: [B, N, F]
        x1 = X.reshape([B, N, 1, F])
        x1 = x1.expand(B, N, self.K, F)
        # x1: [B, N, K, F]

        x2 = KNN_Graph - x1
        # x2: [B, N, K, F]

        x_in = torch.cat([x1, x2], dim=3)
        # x_in: [B, N, K, 2*F]
        x_in = x_in.permute(0, 3, 1, 2)
        # x_in: [B, 2*F, N, K]

        # reshape, x_in: [B, 2*F, N*K]
        x_in = x_in.reshape([B, 2 * F, N * self.K])

        # out: [B, an, N*K]
        out = self.mlp(x_in)
        _, an, _ = out.shape
        # print(out.shape)

        out = out.reshape([B, an, N, self.K])
        # print(out.shape)
        # reshape, out: [B, an, N, K]
        out = out.reshape([B, an*N, self.K])
        # print(out.shape)
        # reshape, out: [B, an*N, K]
        out = nn.MaxPool1d(self.K)(out)
        # print(out.shape)
        out = out.reshape([B, an, N])
        # print(out.shape)
        out = out.permute(0, 2, 1)
        # print(out.shape)

        return out



if __name__ == '__main__':
    inputs = torch.rand((32, 1, 128, 128))
    landmarks = torch.randint(0, 128, (32, 2, 55))

    model = ResNetAndDGCNN(20, num_classes=6)

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