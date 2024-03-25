import math
from pprint import pprint
from warnings import warn

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def make_knn_adj(df: pd.DataFrame, k: int) -> csr_matrix:
    X = df.drop(columns="50K")
    A = kneighbors_graph(X, k)
    assert isinstance(A, csr_matrix)
    return A


def adj_to_graph(A: csr_matrix) -> ig.Graph:
    graph = ig.Graph.Adjacency(A.astype(np.int_))
    return graph


def add_terminate_point(graph: ig.Graph, df: pd.DataFrame) -> list[int]:
    vertices = np.nonzero(df["50K"] == 1)[0].tolist()
    graph.add_vertex("t")
    graph.add_edges(
        [(v, "t") for v in vertices], {"cost": [(0.0, 0.0)] * len(vertices)}
    )
    return vertices


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
    eps = 1e-3
    m = a["workclass"] / (a["hours-per-week"] + eps) - b["workclass"] / (
        b["hours-per-week"] + eps
    )
    payment += 1.0 / (1.0 + np.exp(m))

    # gain and loss
    payment += b["capital-gain"] - a["capital-gain"]
    payment -= b["capital-loss"] - a["capital-loss"]

    return time, payment


def set_cost(graph: ig.Graph, df: pd.DataFrame) -> None:
    for e in graph.es:
        u, v = e.tuple
        e["cost"] = cost(df, u, v)


def merge(
    dists: list[list[tuple[int, float, float]]],
    u: int,
    v: int,
    w: tuple[float, float],
    *,
    limit: int,
) -> None:
    dist_u = dists[u]
    dist_v = dists[v]
    assert isinstance(w, tuple), "not tuple"

    new_v = [(u, child[1] + w[0], child[2] + w[1]) for child in dist_u]
    dist_v += new_v

    dists[v] = dominant_points_2d(dist_v, limit=limit)


def dominant_points_2d(
    points: list[tuple[int, float, float]],
    *,
    limit: int,
) -> list[tuple[int, float, float]]:
    points.sort(key=lambda x: x[1:3])

    res: list[tuple[int, float, float]] = []

    y_min = float("inf")

    for p, x, y in points:
        if y < y_min:
            res.append((p, x, y))
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
) -> list[list[tuple[int, float, float]]]:
    parent_dists: list[list[tuple[int, float, float]]] = [
        [] for _ in range(graph.vcount())
    ]
    parent_dists[source].append((source, 0.0, 0.0))

    # use s-u to u-v to merge s-v
    for _ in range(graph.vcount() - 1):
        for e in graph.es:
            u, v = e.tuple
            merge(parent_dists, u, v, e["cost"], limit=limit)
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
) -> tuple[ig.Graph, list[int], list[list[tuple[int, float, float]]]]:
    adj = make_knn_adj(df, k)
    graph = adj_to_graph(adj)
    set_cost(graph, df)
    ts = add_terminate_point(graph, df)
    parent_dists = multicost_shortest_path(graph, source, limit=limit, verbose=verbose)
    return graph, ts, parent_dists


def backtracking(
    graph: ig.Graph,
    dists: list[list[tuple[int, float, float]]],
    s: int,
    t: int,
) -> list[list[int]]:
    paths = []
    for u, sv1, sv2 in dists[t]:
        path = []
        v = t
        while s != v:
            dist_u = dists[u]
            path.append(u)

            eid = graph.get_eid(u, v)
            uv1, uv2 = graph.es[eid]["cost"]

            for i, si1, si2 in dist_u:
                if math.isclose(si1 + uv1, sv1) and math.isclose(si2 + uv2, sv2):
                    v = u
                    u, sv1, sv2 = i, si1, si2
                    break
            else:
                raise ValueError("No path found!")
        paths.append(path)
    return paths


def get_layout(df: pd.DataFrame) -> list[tuple[int, int]]:
    pca = PCA(2)
    coord: pd.DataFrame = pca.fit_transform(df.drop(columns=["50K"]))  # type: ignore
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
        edge_label=[f"{a:.2f},{b:.2f}" for a, b in graph.es["cost"]],
    )
    plt.show()
