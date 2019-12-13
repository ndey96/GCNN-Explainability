import torch
import torch.nn.functional as F
from data import load_bbbp
from torch_geometric.data import DataLoader
import numpy as np
from models import GCN
import hyperparameters as hp
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

dataset = load_bbbp(hp.N)
random.Random(hp.shuffle_seed).shuffle(dataset)
split_idx = int(np.floor(len(dataset)*hp.train_frac))
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]
train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hp.H_0, hp.H_1, hp.H_2, hp.H_3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
# scheduler = ReduceLROnPlateau(optimizer, 'min')

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
    total_loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            loss = F.binary_cross_entropy(output, data.y)
            total_loss += loss.item() * data.num_graphs
            pred_y = (output >= 0.5).float()
        correct += torch.sum(data.y == pred_y).item()
    test_acc = correct / len(loader.dataset)
    test_loss = total_loss / len(loader.dataset)
    return test_loss, test_acc

train_losses = []
test_losses = []
test_accs = []
for epoch in range(1, hp.epochs + 1):
    loss = train(train_loader)
    train_losses.append(loss)
    test_loss, test_acc = test(test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    # scheduler.step(test_loss)

    if epoch % 10 == 0:
        plt.plot(train_losses)
        plt.title('Loss vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig('loss_vs_epoch')
        plt.close()

        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('train_test_loss_vs_epoch')
        plt.close()

        plt.plot(test_accs)
        plt.title('Test Accuracy vs Epoch')
        plt.ylabel('Test Accuracy')
        plt.xlabel('Epoch')
        plt.savefig('test_acc_vs_epoch')
        plt.close()

torch.save(model.state_dict(), 'gcn_state_dict.pt')

