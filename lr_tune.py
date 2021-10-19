# Based on https://fastai1.fast.ai/callbacks.lr_finder.html

import torch
import torch.nn.functional as F
from data import load_bbbp
from torch_geometric.data import DataLoader
import numpy as np
from models import GCN
import hyperparameters as hp
import random
import matplotlib.pyplot as plt

dataset = load_bbbp(hp.N)
random.Random(hp.shuffle_seed).shuffle(dataset)
split_idx = int(np.floor(len(dataset)*hp.train_frac))
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]
train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hp.H_0, hp.H_1, hp.H_2, hp.H_3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)

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



# increase learning rate from 1e-5 to 1 over 150 epochs
losses = []
test_accs = []
lrs = []
for epoch in range(1, 150 + 1):
    new_lr = 1e-5 * 10**(epoch / 30)
    lrs.append(new_lr)
    optimizer.param_groups[0]['lr'] = new_lr
    loss = train(train_loader)
    losses.append(loss)
    test_acc = test(test_loader)
    test_accs.append(test_acc)
    print(f'Epoch {epoch}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
    # scheduler.step()
    # lr = optimizer.state_dict()['param_groups'][0]['lr']

plt.semilogx(lrs, losses)
plt.ylim(0,1)
plt.title('Learning Rate vs Loss')
plt.ylabel('Loss')
plt.xlabel('Learning Rate')
plt.savefig('lr_vs_loss')
plt.close()

plt.semilogx(lrs, test_accs)
plt.ylim(0,1)
plt.title('Learning Rate vs Test Accuracy')
plt.ylabel('Test Accuracy')
plt.xlabel('Learning Rate')
plt.savefig('lr_vs_acc')
plt.close()
