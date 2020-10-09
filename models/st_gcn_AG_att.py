import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from .utils.tgcn import ConvTemporalGraphical
# from .utils.graph import Graph
from .utils.tgcn import ConvTemporalGraphical
from .utils.graph import Graph

class ModelAGATT(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels_g, in_channels_h, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        self.model_g = Model_G(in_channels_g, num_class, graph_args, edge_importance_weighting, **kwargs)
        self.model_h = Model_A(in_channels_h, num_class, graph_args, edge_importance_weighting, **kwargs)
        graph_size = self.model_g.A.shape[-1] * 1 * num_class
        hog_size = self.model_h.A.shape[-1] * 1 * num_class
        self.drop = nn.Dropout(0.5, inplace=True)
        self.fc1 = nn.Linear(hog_size + graph_size, 256)
        self.fc2 = nn.Linear(256, num_class)
        # self.fc = nn.Linear(hog_size + graph_size, num_class)
        self.alpha_g = nn.Sequential(nn.Linear(64, 1),
                        nn.Sigmoid())
        
        self.alpha_h = nn.Sequential(nn.Linear(128, 1),
                        nn.Sigmoid())

    def forward(self, x_g, x_h):

        out_g = self.model_g(x_g)
        out_h = self.model_h(x_h)

        out_g = torch.mean(out_g, dim=2, keepdim=True)
        out_h = torch.mean(out_h, dim=2, keepdim=True)
        out_g = out_g.view(out_g.size()[0], -1)
        out_h = out_h.view(out_h.size()[0], -1)
        out = torch.cat([out_g, out_h], dim=1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        # out = self.fc(out)
        return out

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

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


class Model_G(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
        ))
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)
        # self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        # N, C, T, V, M = x.size()
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        x = x.view(N, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        # x = F.avg_pool2d(x, x.size()[2:])
        # x = x.view(N, -1, 1, 1)

        # prediction
        x = self.fcn(x)
        # x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

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

class Model_A(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        #     st_gcn(256, 256, kernel_size, 1, **kwargs),
        # ))

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs),
        ))
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        # self.fcn = nn.Conv2d(64, num_class, kernel_size=1)
        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        # N, C, T, V, M = x.size()
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        x = x.view(N, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        # x = F.avg_pool2d(x, x.size()[2:])
        # x = x.view(N, -1, 1, 1)

        # prediction
        x = self.fcn(x)
        # x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

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


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.t_att = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

        self.s_att = ConvTemporalGraphical(1, 1, kernel_size[1])
        self.sigmoid = nn.Sigmoid()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        # attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        s_out, A = self.s_att(avg_out, A) 
        s_out = self.sigmoid(s_out)

        t_x = self.tcn(x)
        t_out = self.t_att(t_x)

        t_x = t_x * s_out * t_out
        x = t_x + res

        return self.relu(x), A
    

if __name__ == '__main__':
    inputs_g = torch.rand((16, 2, 3, 55))
    inputs_h = torch.rand((16, 36, 3, 55))
    model = ModelAG(2, 36, 6, {}, False)
    outputs = model(inputs_g, inputs_h)
    print(outputs)