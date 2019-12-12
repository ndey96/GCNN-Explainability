import torch
import torch.nn.functional as F
from data import load_bbbp
import numpy as np
from models import GCN
import matplotlib.pyplot as plt
import hyperparameters as hp
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

dataset = load_bbbp(hp.N)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(hp.H_0, hp.H_1, hp.H_2, hp.H_3).to(device)
model.load_state_dict(torch.load('gcn_state_dict.pt'))
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(model)
breakpoint()
model.train()
total_loss = 0
for data in [dataset[0]]:
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = F.binary_cross_entropy(out, data.y)
    loss.backward()

    input_grads = model.input.grad.view(32, 40, 8)
    print(saliency_map(input_grads))

    final_conv_acts = model.final_conv_acts.view(32, 40, 512)
    final_conv_grads = model.final_conv_grads.view(32, 40, 512)
    print(grad_cam(final_conv_acts, final_conv_grads))
    # breakpoint()
    total_loss += loss.item() * data.num_graphs
