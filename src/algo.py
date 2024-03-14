import math
from typing import Mapping
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
    time = 0.0
    payment = 0.0
    a: pd.Series[float] = df.loc[i]  # type: ignore
    b: pd.Series[float] = df.loc[j]  # type: ignore

    # for age
    time = max(time, b["age"] - a["age"])

    # education
    time = max(time, b["education-num"] - a["education-num"])

    # workclass
    time = max(time, b["workclass"] - a["workclass"])

    # sigmoid(workclass : hours-per-week)
    m = a["workclass"] / a["hours-per-week"] - b["workclass"] / b["hours-per-week"]
    payment += 1.0 / (1.0 + math.exp(m))

    # gain and loss
    payment += b["capital-gain"] - a["capital-gain"]
    payment -= b["capital-loss"] - a["capital-loss"]

    return time, payment


def merge(
    cost: tuple[float, float],
    dists: list[list[tuple[float, float]]],
    i: int,
    j: int,
) -> None:
    u = dists[i]
    v = dists[j]
