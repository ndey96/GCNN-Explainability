import torch
import torch.nn.functional as F
from data import load_bbbp
from torch_geometric.data import DataLoader
import numpy as np
from models import GCN
import hyperparameters as hp
import random

dataset = load_bbbp(hp.N)
random.Random(hp.shuffle_seed).shuffle(dataset)
split_idx = int(np.floor(len(dataset)*hp.train_frac))
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]
train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hp.H_0, hp.H_1, hp.H_2, hp.H_3).to(device)
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


for epoch in range(1, hp.epochs + 1):
    loss = train(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch {epoch}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
    scheduler.step()
    # lr = optimizer.state_dict()['param_groups'][0]['lr']

torch.save(model.state_dict(), 'gcn_state_dict.pt')