import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat=8, h1=128, h2=256, h3=512, nclass=2):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, h1)
        self.gc2 = GraphConvolution(h1, h2)
        self.gc3 = GraphConvolution(h2, h3)

    def forward(self, X, A):
        h1 = F.relu(self.gc1(X, A))
        h2 = F.relu(self.gc2(h1, A))
        h3 = F.relu(self.gc3(h2, A))
        h4 = F.avg_pool1d(h3)
        return F.sigmoid(h4, dim=1)
