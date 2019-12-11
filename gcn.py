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
        self.conv1 = GCNConv(H_0, H_1)
        self.conv2 = GCNConv(H_1, H_2)
        self.conv3 = GCNConv(H_2, H_3)
        self.fc1 = torch.nn.Linear(H_3, 1)

    def forward(self, data):
        h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        h1 = F.relu(self.conv1(h0, edge_index, edge_weight))
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h3 = F.relu(self.conv3(h2, edge_index, edge_weight))
        h4 = global_mean_pool(h3, data.batch)
        out = torch.nn.Sigmoid()(self.fc1(h4))
        breakpoint()
        # print(h0.shape)
        # print(h1.shape)
        # print(h2.shape)
        # print(h3.shape)
        # print(h4.shape)
        # print(out.shape)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(H_0, H_1, H_2, H_3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(model)

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward()
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
