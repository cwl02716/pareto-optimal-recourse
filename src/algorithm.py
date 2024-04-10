import math
from itertools import product
from typing import Callable
from warnings import warn

import igraph as ig
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


def add_pseudo_target(graph: ig.Graph, vs: list[int]) -> None:
    t = graph.add_vertex().index
    graph.add_edges([(v, t) for v in vs], {"costs": [[(0.0, 0.0)] for _ in vs]})


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
    graph: ig.Graph, source: int, *, limit: int
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
    return parent_dists


def recourse(
    X: pd.DataFrame,
    k: int,
    source: int,
    terminates: list[int],
    cost_fn: Callable[[int, int], list[tuple[float, float]]],
    *,
    limit: int,
    verbose: bool = False,
) -> tuple[
    ig.Graph,
    list[list[tuple[tuple[int, int], tuple[float, float]]]],
]:
    if verbose:
        print("Starting recourse algorithm...")
    adj: csr_matrix = kneighbors_graph(X, k)  # type: ignore
    graph = ig.Graph.Adjacency(adj.astype(np.int_))

    for e in graph.es:
        u, v = e.tuple
        e["costs"] = cost_fn(u, v)

    add_pseudo_target(graph, terminates)
    parent_dists = multicost_shortest_path(graph, source, limit=limit)
    if verbose:
        print("Recourse algorithm finished!")
    return graph, parent_dists


def backtracking(
    graph: ig.Graph,
    dists: list[list[tuple[tuple[int, int], tuple[float, float]]]],
    s: int,
    t: int,
    *,
    verbose: bool = False,
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
        path.reverse()
        paths.append(path)
        if verbose:
            print(f"Path: {path}")
    return paths
