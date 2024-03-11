import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
import igraph as ig


def make_knn_adj(df: pd.DataFrame, k: int) -> csr_matrix:
    X = df.drop(columns="50K")
    A = kneighbors_graph(X, k)
    assert isinstance(A, csr_matrix)
    return A


def ajd_to_graph(A: csr_matrix) -> pd.DataFrame:
    graph = ig.Graph.Adjacency(A.astype(np.int_))
    return graph

