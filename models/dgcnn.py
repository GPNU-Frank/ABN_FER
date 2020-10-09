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


class DGCNN(nn.Module):

    def __init__(self, num_classes=1000, gcn_hidden=512, gcn_out=64, dropout=0.5):
        super(DGCNN, self).__init__()

        self.edge_conv_1 = EdgeConv(layers=[2, 64, 64, 64])
        # self.edge_conv_2 = EdgeConv(layers=[64, 128])
        self.conv_block = conv_bn_block(64, 64, 1)
        # self.fc_block = fc_bn_block(1024, 512)
        self.fc = nn.Linear(64, num_classes)


    def forward(self, landmark):

        # landmark = landmark.permute(0, 2, 1)
        landmark = landmark.permute(0, 2, 3, 1)
        # B, N, C = landmark.shape
        B, T, N, C = landmark.shape

        x = self.edge_conv_1(landmark)
        # x = self.edge_conv_2(x)
        # x = x.permute(0, 2, 1)
        x = x.permute(0, 3, 1, 2)
        # x = self.conv_block(x)
        # x = nn.AvgPool1d(N)(x)
        # x = nn.AvgPool2d((T, N))(x)
        x = nn.MaxPool2d((T, N))(x)
        x = x.squeeze(3).squeeze(2)
        # x = self.fc_block(x)
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


# def conv_bn_block(input, output, kernel_size):
#     '''
#     标准卷积块（conv + bn + relu）
#     :param input: 输入通道数
#     :param output: 输出通道数
#     :param kernel_size: 卷积核大小
#     :return:
#     '''
#     return nn.Sequential(
#         # nn.Conv1d(input, output, kernel_size),
#         nn.Conv2d(input, output, kernel_size),
#         # nn.BatchNorm1d(output),
#         nn.BatchNorm2d(output),
#         nn.ReLU(inplace=True)
#     )

def conv_bn_block(input, output, kernel_size, T=3, padding=(1, 0)):
    '''
    标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    '''
    return nn.Sequential(
        # nn.Conv1d(input, output, kernel_size),
        nn.Conv2d(input, output, kernel_size),
        # nn.BatchNorm1d(output),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True),
        nn.Conv2d(output, output, (T, 1), padding=padding),
        nn.BatchNorm2d(output),
        nn.ReLU(inplace=True),

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
    def __init__(self, layers, K=3, T=3):
        '''
        构造函数
        :param layers: e.p. [3, 64, 64, 64]
        :param K:
        '''
        super(EdgeConv, self).__init__()

        self.K = K
        self.T = T
        self.layers = layers
        # self.KNN_Graph = torch.zeros(Args.batch_size, 2048, self.K, self.layers[0]).to(Args.device)

        if layers is None:
            self.mlp = None
        else:
            mlp_layers = OrderedDict()
            for i in range(len(self.layers) - 1):
                if i == 0:
                    # mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(2*self.layers[i], self.layers[i + 1], 1)
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(2*self.layers[i], self.layers[i + 1], (1, 1))
                else:
                    # mlp_layers['conv_bn_block_{}'.format(i+1)] = conv_bn_block(self.layers[i], self.layers[i+1], 1)
                    mlp_layers['conv_bn_block_{}'.format(i+1)] = conv_bn_block(self.layers[i], self.layers[i+1], (1, 1))
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
        # B, N, F = X.shape
        B, T, N, F = X.shape
        assert F == self.layers[0]

        # KNN_Graph = torch.zeros(B, N, self.K, self.layers[0]).cuda()
        KNN_Graph = torch.zeros(B, self.T, N, self.K, self.layers[0]).cuda()
        # KNN_Graph = torch.zeros(B, N, self.K, self.layers[0])

        # creating knn graph
        # X: [B, N, F]
        for idx, x in enumerate(X):
            # x: [N, F]
            for idt, t_x in enumerate(x):
                KNN_Graph[idx][idt] = self.createSingleKNNGraph(t_x)
            # knn_graph: [N, K, F]
            # self.KNN_Graph[idx] = self.createSingleKNNGraph(x)
            # KNN_Graph[idx] = self.createSingleKNNGraph(x)
        # print(self.KNN_Graph.shape)
        # print('KNN_Graph: {}'.format(KNN_Graph[0][0]))


        # KNN_Graph: B N T K F
        # X: B T N F
        # X: [B, N, F]
        # x1 = X.reshape([B, N, 1, F])
        x1 = X.reshape([B, T, N, 1, F])
        # x1 = x1.expand(B, N, self.K, F)
        x1 = x1.expand(B, T, N, self.K, F)
        # x1: [B, N, K, F]

        x2 = KNN_Graph - x1
        # x2: [B, N, K, F]

        # x_in = torch.cat([x1, x2], dim=3)
        x_in = torch.cat([x1, x2], dim=4)
        # x_in: [B, N, K, 2*F]
        # x_in = x_in.permute(0, 3, 1, 2)
        x_in = x_in.permute(0, 4, 2, 1, 3)  # ori: B, T, N, K, F  aft: B, 2 * F, N, T, K
        # x_in: [B, 2*F, N, K]

        # reshape, x_in: [B, 2*F, N*K]
        # x_in = x_in.reshape([B, 2 * F, N * self.K])
        x_in = x_in.reshape([B, 2 * F, T, N * self.K])

        # out: [B, an, N*K]
        out = self.mlp(x_in)
        _, an, _, _ = out.shape
        # print(out.shape)

        # out = out.reshape([B, an, N, self.K])
        out = out.reshape([B, an, T, N, self.K])
        # print(out.shape)
        # reshape, out: [B, an, N, K]
        out = out.permute(0, 2, 1, 3, 4)
        out = out.reshape([B, T*an*N, self.K])
        # print(out.shape)
        # reshape, out: [B, an*N, K]
        # out = nn.MaxPool1d(self.K)(out)
        out = nn.MaxPool1d(self.K)(out)
        # print(out.shape)
        # out = out.reshape([B, an, N])
        out = out.reshape([B, T, an, N])
        # print(out.shape)
        # out = out.permute(0, 2, 1)
        out = out.permute(0, 1, 3, 2)
        # print(out.shape)

        # B, T, N, F = X.shape
        return out



if __name__ == '__main__':
    inputs = torch.rand((32, 1, 128, 128))
    landmarks = torch.randint(0, 128, (32, 2, 55))
    landmarks = torch.randint(0, 128, (32, 2, 3, 55))

    model = DGCNN(num_classes=6)


    outs = model(landmarks)
    print(outs)