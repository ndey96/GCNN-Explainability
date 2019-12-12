import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool
from torch.autograd import Variable

class GCN(torch.nn.Module):
    def __init__(self, H_0, H_1, H_2, H_3):
        super(GCN, self).__init__()
        # explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        # layers
        self.conv1 = GCNConv(H_0, H_1)
        self.conv2 = GCNConv(H_1, H_2)
        self.conv3 = GCNConv(H_2, H_3)
        self.fc1 = torch.nn.Linear(H_3, 1)

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h0.requires_grad = True
        self.input = h0
        h1 = F.relu(self.conv1(h0, edge_index, edge_weight))
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        with torch.enable_grad():
            self.final_conv_acts = self.conv3(h2, edge_index, edge_weight)
        self.final_conv_acts.register_hook(self.activations_hook)
        h3 = F.relu(self.final_conv_acts)
        h4 = global_mean_pool(h3, data.batch)
        out = torch.nn.Sigmoid()(self.fc1(h4))
        # print(h0.shape)
        # print(h1.shape)
        # print(h2.shape)
        # print(h3.shape)
        # print(h4.shape)
        # print(out.shape)
        return out