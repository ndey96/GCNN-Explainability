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
    df = pd.read_csv('bbbp/BBBP.csv')
    feature_matrices = []  # np.zeros((len(df), N, 1))
    adj_matrices = []  # np.zeros((len(df), N, N))
    labels = []  # np.zeros((len(df), 1))
    smiles_list = []
    nums = []
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        nums.append(row.num)
        smiles_list.append(row.smiles)
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
    one_hot_feature_matrices = enc.fit_transform(feature_matrices)
    one_hot_feature_matrices = np.reshape(one_hot_feature_matrices, (-1, N, 8))

    dataset = []
    for i in range(len(labels)):
        X = torch.from_numpy(one_hot_feature_matrices[i]).float()
        A = torch.from_numpy(adj_matrices[i]).float()
        y = torch.Tensor([[labels[i]]]).float()
        mol_num = torch.Tensor([nums[i]])
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        # breakpoint()
        dataset.append(Data(x=X,
                            edge_index=edge_index,
                            edge_attr=edge_weight,
                            y=y,
                            # smiles=smiles_list[i],
                            A=A,
                            # atomic_nums=feature_matrices[i],
                            mol_num=mol_num
                            ))

    return dataset
