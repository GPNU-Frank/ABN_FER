import torch
import torch.nn as nn
import math
from torch.nn import Module
from torch.nn.parameter import Parameter

from .utils.graph import Graph
from .utils.tgcn import ConvTemporalGraphical

import torch.nn.functional as F




class GCNPair(Module):

    def __init__(self, numclasses=6):
        super(GCNPair, self).__init__()

        graph = Graph({})
        self.A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.A = self.A.cuda()

        self.gcn11 = ConvTemporalGraphical(9, 64, 1)
        self.bn11 = nn.BatchNorm2d(64)
        self.gcn12 = ConvTemporalGraphical(64, 128, 1)
        self.bn12 = nn.BatchNorm2d(128)
        self.gcn13 = ConvTemporalGraphical(128, 256, 1)
        self.bn13 = nn.BatchNorm2d(256)

        self.gcn21 = ConvTemporalGraphical(9, 64, 1)
        self.bn21 = nn.BatchNorm2d(64)
        self.gcn22 = ConvTemporalGraphical(64, 128, 1)
        self.bn22 = nn.BatchNorm2d(128)
        self.gcn23 = ConvTemporalGraphical(128, 256, 1)
        self.bn23 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512, numclasses)
        self.bn_f1 = nn.BatchNorm1d(256) 
        # self.drop_f1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, numclasses)

        self.tensor_network = TenorNetworkModule()

    def forward(self, landmark):
        x1 = landmark[:, :, :, :, 0]  # batch_size, channel, 1, numbers, frame 
        x2 = landmark[:, :, :, :, 1]

        # g1 = edge[]  # batch_size, edge_num, 2, frame

        x1, _ = self.gcn11(x1, self.A)
        x1 = self.bn11(x1)
        x1 = self.relu(x1)

        x1, _ = self.gcn12(x1, self.A)
        x1 = self.bn12(x1)
        x1 = self.relu(x1)

        x1, _ = self.gcn13(x1, self.A)
        x1 = self.bn13(x1)
        x1 = self.relu(x1)

        x1 = F.adaptive_avg_pool2d(x1, 1)
        x1 = x1.squeeze(-1).squeeze(-1)

        x2, _ = self.gcn21(x2, self.A)
        x2 = self.bn21(x2)
        x2 = self.relu(x2)

        x2, _ = self.gcn22(x2, self.A)
        x2 = self.bn22(x2)
        x2 = self.relu(x2)

        x2, _ = self.gcn23(x2, self.A)
        x2 = self.bn23(x2)
        x2 = self.relu(x2)

        x2 = F.adaptive_avg_pool2d(x2, 1)
        x2 = x2.squeeze(-1).squeeze(-1)


        # t_n = self.tensor_network(x1, x2)
        res = torch.cat([x1, x2], dim=1)

        res = self.fc1(res)
        # res = self.bn_f1(res)
        # res = self.relu(res)

        # res = self.fc2(res)

        return res
    
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


class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(256,
                                                             256,
                                                             16))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(16,
                                                                   2 * 256))
        self.bias = torch.nn.Parameter(torch.Tensor(16, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(256, -1))
        scoring = scoring.view(256, 16)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


if __name__ == '__main__':
    inputs = torch.rand((32, 9, 1, 55, 2))

    model = GCNPair()

    outputs = model(inputs)

    print(outputs)