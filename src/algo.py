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


def cost(df: pd.DataFrame, i: int, j: int) -> tuple[float, float]:
    time = 0
    payment = 0
    # for age
    if (df.loc[j, "age"] - df.loc[i, "age"]>= 0):
        time.max(time, df.loc[j, "age"] - df.loc[i, "age"]) 
   
    # education
    if (df.loc[j, "education-num"] - df.loc[i, "education-num"]>= 0):
        time.max(time, df.loc[j, "education-num"] - df.loc[i, "education-num"])

    # hours-per-week and workclass
    t = df.loc[j, "workclass"] - df.loc[i, "workclass"]
    time.max(time, t) if t > 0 else time.max(time, 0)
    p = df.loc[j, "hours-per-week"] - df.loc[i, "hours-per-week"]/t
    # do sigmoid to p
    payment += 1/(1+np.exp(-p))

    # gain and loss
    payment += (df.loc[j, "capital-gain"] - df.loc[i, "capital-gain"]) + (df.loc[i, "capital-loss"] - df.loc[j, "capital-loss"])


def merge(i: int, j: int): ...
