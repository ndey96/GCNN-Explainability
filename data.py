from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.data import Data
import torch
from tqdm import tqdm

def load_bbbp(N=40):
    print('Loading data...')
    df = pd.read_csv('bbbp/BBBP.csv', index_col='num')
    feature_matrices = []  # np.zeros((len(df), N, 1))
    adj_matrices = []  # np.zeros((len(df), N, N))
    labels = []  # np.zeros((len(df), 1))
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            continue

        # Adjacency Matrix
        adj = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        adj_matrix = np.zeros((N, N))
        s0, s1 = adj.shape
        if s0 > N:
            continue
        # adj_matrix[:s0, :s1] = adj + np.eye(s0)
        adj_matrix[:s0, :s1] = adj
        adj_matrices.append(adj_matrix)


        # Feature Vector
        atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        padded_atomic_nums = [0] * N
        padded_atomic_nums[:len(atomic_nums)] = atomic_nums
        feature_matrices.append(padded_atomic_nums)

        # Labels
        labels.append(row.p_np)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    feature_matrices = enc.fit_transform(feature_matrices)
    feature_matrices = np.reshape(feature_matrices, (-1, N, 8))

    dataset = []
    for i in range(len(labels)):
        X = torch.from_numpy(feature_matrices[i]).float()
        A = adj_matrices[i]
        y = torch.Tensor(labels[i]).float()
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        dataset.append(Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y))

    return dataset
