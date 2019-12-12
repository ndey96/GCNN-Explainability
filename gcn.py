import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool
from data import load_bbbp
from torch_geometric.data import DataLoader
import numpy as np
from random import shuffle
# parser = argparse.ArgumentParser()
# parser.add_argument('--use_gdc', action='store_true',
#                     help='Use GDC preprocessing.')
# args = parser.parse_args()

epochs = 200
N = 40
H_0 = 8
H_1 = 128
H_2 = 256
H_3 = 512
train_frac = 0.8
# dataset = TUDataset(root='/tmp/Tox21_AR', name='Tox21_AR')
dataset = load_bbbp(N)
shuffle(dataset)
split_idx = int(np.floor(len(dataset)*train_frac))
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self, H_0, H_1, H_2, H_3):
        super(Net, self).__init__()
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
        print('hook emmmmmm')

    def forward(self, data):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h0.requires_grad = True
        self.input = h0
        h1 = F.relu(self.conv1(h0, edge_index, edge_weight))
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        a3 = self.conv3(h2, edge_index, edge_weight)
        a3.register_hook(self.activations_hook)
        self.final_conv_acts = a3
        h3 = F.relu(a3)
        # h3.requires_grad = True
        self.final_conv = h3
        h4 = global_mean_pool(h3, data.batch)
        out = torch.nn.Sigmoid()(self.fc1(h4))
        print(h0.shape)
        print(h1.shape)
        print(h2.shape)
        print(h3.shape)
        print(h4.shape)
        print(out.shape)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(H_0, H_1, H_2, H_3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(model)

def saliency_map(input_grads):
    for i in range(input_grads.shape[0]):
        node_saliency_map = []
        for n in range(input_grads.shape[1]):
            node_grads = input_grads[i,n,:]
            node_saliency = torch.norm(F.relu(node_grads)).item()
            node_saliency_map.append(node_saliency)
    return node_saliency_map

def grad_cam(final_conv_acts, final_conv_grads):
    for i in range(final_conv_grads.shape[0]): # ith example in batch
        node_heat_map = []
        alphas = torch.mean(final_conv_grads[i], axis=0) # mean gradient for each feature (512x1)
        for n in range(final_conv_grads.shape[1]): # nth node
            node_heat = (alphas @ final_conv_acts[i, 0]).item()
            node_heat_map.append(node_heat)

    return node_heat_map


def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward()

        input_grads = model.input.grad.view(32, 40, 8)
        print(saliency_map(input_grads))

        final_conv_acts = model.final_conv_acts.view(32, 40, 512)
        final_conv_grads =  model.final_conv_grads.view(32,40,512)
        print(grad_cam(final_conv_acts, final_conv_grads))
        breakpoint()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            pred_y = (output >= 0.5).float()

        correct += torch.sum(data.y == pred_y).item()
    return correct / len(loader.dataset)


for epoch in range(1, epochs + 1):
    loss = train(train_loader)
    test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()
