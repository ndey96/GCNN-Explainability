from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.data import Data

def load_data(N=40):
    df = pd.read_csv('bbbp/BBBP.csv', index_col='num')
    feature_matrices = []  # np.zeros((len(df), N, 1))
    adj_matrices = []  # np.zeros((len(df), N, N))
    labels = []  # np.zeros((len(df), 1))
    for i in range(len(df)):
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
        X = feature_matrices[i]
        A = adj_matrices[i]
        y = labels[i]
        A_coo = coo_matrix(A)
        edge_index = np.vstack([A_coo.row, A_coo.col])
        edge_weight = A_coo.data
        dataset.append(Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y))

    return dataset
