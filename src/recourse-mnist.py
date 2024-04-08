import math
from itertools import product
from pprint import pprint
from warnings import warn

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper import YCOL
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


train_samples = 5000

# Load data from https://www.openml.org/d/554
X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False
)  # keep numpy.ndarray
scaler = StandardScaler()


def make_knn_adj(df: pd.DataFrame, k: int) -> csr_matrix:
    X = df.drop(columns=YCOL)
    A = kneighbors_graph(X, k)
    assert isinstance(A, csr_matrix)
    return A


def adj_to_graph(A: csr_matrix) -> ig.Graph:
    graph = ig.Graph.Adjacency(A.astype(np.int_))
    return graph


def add_terminate_point(graph: ig.Graph, df: pd.DataFrame) -> list[int]:
    vertices = np.nonzero(df[YCOL] == 1)[0].tolist()
    graph.add_vertex("t")
    graph.add_edges(
        [(v, "t") for v in vertices], {"costs": [[(0.0, 0.0)] for _ in vertices]}
    )
    return vertices


def costs(df: np.ndarray, i: int, j: int) -> list[tuple[float, float]]:
    d1 = 0.0  # l1 distance
    d2 = 0.0  # l2 distance

    a: pd.Series[float] = df.iloc[i]  # type: ignore
    b: pd.Series[float] = df.iloc[j]  # type: ignore
    m: pd.Series[float] = df.iloc[[i, j]]  # type: ignore
    diff = a - b
    d1 = np.linalg.norm(ord=1, x=diff)
    d2 = np.linalg.norm(ord=2, x=diff)
    # 取最大值of m
    m0 = m.max(axis=1)  # type: ignore

    d2 = math.sqrt(d2.sum())
    d1 /= m0
    d2 /= m0
    return [(d1, d2)]  # type: ignore


def set_costs(graph: ig.Graph, df: pd.DataFrame) -> None:
    for e in graph.es:
        u, v = e.tuple
        e["costs"] = costs(df, u, v)


def merge(
    dists: list[list[tuple[tuple[int, int], tuple[float, float]]]],
    u: int,
    v: int,
    costs: list[tuple[float, float]],
    *,
    limit: int,
) -> None:
    dist_u = dists[u]
    dist_v = dists[v]

    for (_, cost_u), (i, cost_w) in product(dist_u, enumerate(costs)):
        dist_v += [((u, i), (cost_u[0] + cost_w[0], cost_u[1] + cost_w[1]))]

    dists[v] = dominant_points_2d(dist_v, limit=limit)


def dominant_points_2d(
    points: list[tuple[tuple[int, int], tuple[float, float]]],
    *,
    limit: int,
) -> list[tuple[tuple[int, int], tuple[float, float]]]:
    points.sort(key=lambda x: x[1])

    res: list[tuple[tuple[int, int], tuple[float, float]]] = []

    y_min = float("inf")

    for p, (x, y) in points:
        if y < y_min:
            res.append((p, (x, y)))
            y_min = y

    size = len(res)
    if size > limit:
        warn(f"Exceed limit {size}!")
        step = (size - 1) / (limit - 1)
        res = [res[round(i * step)] for i in range(limit)]

    return res


def multicost_shortest_path(
    graph: ig.Graph,
    source: int,
    *,
    limit: int,
    verbose: bool = False,
) -> list[list[tuple[tuple[int, int], tuple[float, float]]]]:
    parent_dists: list[list[tuple[tuple[int, int], tuple[float, float]]]] = [
        [] for _ in range(graph.vcount())
    ]
    parent_dists[source].append(((source, 0), (0.0, 0.0)))

    # use s-u to u-v to merge s-v
    for _ in range(graph.vcount() - 1):
        for e in graph.es:
            u, v = e.tuple
            merge(parent_dists, u, v, e["costs"], limit=limit)
        if verbose:
            pprint(parent_dists, indent=4)
    return parent_dists


def recourse(
    df: pd.DataFrame,
    k: int,
    source: int,
    *,
    limit: int,
    verbose: bool = False,
) -> tuple[
    ig.Graph,
    list[int],
    list[
        list[
            tuple[
                tuple[int, int],
                tuple[float, float],
            ]
        ]
    ],
]:
    adj = make_knn_adj(df, k)
    graph = adj_to_graph(adj)
    set_costs(graph, df)
    ts = add_terminate_point(graph, df)
    parent_dists = multicost_shortest_path(graph, source, limit=limit, verbose=verbose)
    return graph, ts, parent_dists


def backtracking(
    graph: ig.Graph,
    dists: list[list[tuple[tuple[int, int], tuple[float, float]]]],
    s: int,
    t: int,
) -> list[list[int]]:
    paths = []
    for (u, w), (sv1, sv2) in dists[t]:
        path = []
        v = t
        while s != v:
            dist_u = dists[u]
            path.append(u)

            eid = graph.get_eid(u, v)
            uv1, uv2 = graph.es[eid]["costs"][w]

            for (i, j), (si1, si2) in dist_u:
                if math.isclose(si1 + uv1, sv1) and math.isclose(si2 + uv2, sv2):
                    v = u
                    u, w = i, j
                    sv1, sv2 = si1, si2
                    break
            else:
                raise ValueError("No path found!")
        paths.append(path)
    return paths


def get_layout(df: pd.DataFrame) -> list[tuple[int, int]]:
    pca = PCA(2)
    coord: pd.DataFrame = pca.fit_transform(df.drop(columns=[YCOL]))  # type: ignore
    return coord.to_numpy().tolist()


def show_graph(
    graph: ig.Graph, coord: list[tuple[int, int]], source: int, ts: list[int]
):
    fig, ax = plt.subplots(figsize=(12, 12), layout="tight")
    ig.plot(
        graph,
        ax,
        layout=coord,
        # edge_arrow_size=2,
        vertex_label=graph.vs.indices,
        vertex_color=[
            "red" if i == source else "blue" if i in ts else "lightblue"
            for i in range(graph.vcount())
        ],
        edge_label=[f"{a:.2f},{b:.2f}" for (a, b), *_ in graph.es["costs"]],
    )
    plt.show()
