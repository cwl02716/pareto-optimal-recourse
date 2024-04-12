import math
from typing import Callable
from warnings import warn

import igraph as ig
import pandas as pd
from sklearn.neighbors import kneighbors_graph

type CostT = tuple[float, float]
type ParentAndCostT = tuple[int, CostT]


def add_fake_target(graph: ig.Graph, vs: list[int], *, key: str) -> None:
    t = graph.add_vertex().index
    graph.add_edges([(v, t) for v in vs], {key: [(0.0, 0.0) for _ in vs]})


def merge(
    dist_list: list[list[ParentAndCostT]],
    u: int,
    v: int,
    cost_uv: CostT,
    limit: int,
    *,
    verbose: bool,
) -> None:
    dist_u = dist_list[u]
    dist_v = dist_list[v]
    dist_v += [
        (u, (cost_su[0] + cost_uv[0], cost_su[1] + cost_uv[1])) for _, cost_su in dist_u
    ]
    dist_list[v] = dominant_points_2d(dist_v, limit, verbose=verbose)


def dominant_points_2d(
    dist: list[ParentAndCostT],
    limit: int,
    *,
    verbose: bool,
) -> list[ParentAndCostT]:
    dist.sort(key=lambda x: x[1])
    new_dist = []

    y_min = math.inf
    for p, (x, y) in dist:
        if y < y_min:
            new_dist.append((p, (x, y)))
            y_min = y

    size = len(new_dist)

    if size > limit:
        if verbose:
            warn(f"Exceed limit {size}!")

        step = (size - 1) / (limit - 1)
        new_dist = [new_dist[round(i * step)] for i in range(limit)]

    return new_dist


def multicost_shortest_paths(
    graph: ig.Graph,
    source: int,
    limit: int,
    *,
    key: str,
    verbose: bool,
) -> list[list[ParentAndCostT]]:
    dist_list = [[] for _ in range(graph.vcount())]
    dist_list[source].append((source, (0.0, 0.0)))

    for _ in range(graph.vcount() - 1):
        for e in graph.es:
            u, v = e.tuple
            merge(dist_list, u, v, e[key], limit, verbose=verbose)
    return dist_list


def make_knn_graph(X: pd.DataFrame, k: int) -> ig.Graph:
    adj = kneighbors_graph(X, k)
    graph = ig.Graph.Adjacency(adj.toarray())  # type: ignore
    return graph


def recourse(
    graph: ig.Graph,
    source: int,
    targets: list[int],
    cost_fn: Callable[[int, int], CostT],
    limit: int,
    *,
    key: str,
    verbose: bool,
) -> list[list[ParentAndCostT]]:
    if verbose:
        print("Starting recourse algorithm...")

    for e in graph.es:
        u, v = e.tuple
        e[key] = cost_fn(u, v)

    add_fake_target(graph, targets, key=key)
    parent_dists = multicost_shortest_paths(
        graph, source, limit, key=key, verbose=verbose
    )

    if verbose:
        print("Recourse algorithm finished!")

    return parent_dists


def backtracking(
    graph: ig.Graph,
    dist_list: list[list[ParentAndCostT]],
    s: int,
    t: int,
    *,
    key: str,
    verbose: bool,
) -> list[list[int]]:
    paths = []
    for u, (sv1, sv2) in dist_list[t]:
        path = []
        v = t
        while s != v:
            dist_u = dist_list[u]
            path.append(u)

            eid = graph.get_eid(u, v)
            uv1, uv2 = graph.es[eid][key]

            for i, (si1, si2) in dist_u:
                if math.isclose(si1 + uv1, sv1) and math.isclose(si2 + uv2, sv2):
                    v = u
                    u, sv1, sv2 = i, si1, si2
                    break
            else:
                raise ValueError("No path found!")
        path.reverse()
        paths.append(path)

    if verbose:
        for i, ((_, (c1, c2)), path) in enumerate(zip(dist_list[t], paths)):
            print(f"Path {i} ({c1:.2f}, {c2:.2f}): {path}")

    return paths
