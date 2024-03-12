import igraph as ig
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


def make_knn_adj(df: pd.DataFrame, k: int) -> csr_matrix:
    X = df.drop(columns="50K")
    A = kneighbors_graph(X, k)
    assert isinstance(A, csr_matrix)
    return A


def adj_to_graph(A: csr_matrix) -> pd.DataFrame:
    graph = ig.Graph.Adjacency(A.astype(np.int_))
    return graph


def cost(df: pd.DataFrame, i: int, j: int) -> tuple[float, float]: ...


def merge(i: int, j: int): ...
