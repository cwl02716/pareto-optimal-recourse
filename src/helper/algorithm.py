from __future__ import annotations

import math
from abc import abstractmethod
from operator import itemgetter
from typing import Callable, Protocol, Self, SupportsIndex
from warnings import warn

import igraph as ig
import pandas as pd
from sklearn.neighbors import kneighbors_graph


class Comparable(Protocol):
    def __eq__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Self, /) -> bool: ...


class Cost[T: Comparable](Protocol):
    value: T

    def __init__(self, value: T) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other: Self) -> bool:
        return self.value < other.value

    def __repr__(self) -> str:
        return repr(self.value)

    @abstractmethod
    def __add__(self, other: Self) -> Self: ...

    @abstractmethod
    def lowerbound(self) -> Self: ...

    @abstractmethod
    def upperbound(self) -> Self: ...

    @abstractmethod
    def identity(self) -> Self: ...


class FloatingCost(Cost[float]):
    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return format(self.value, ".4g")

    def isclose(self, other: Self) -> bool:
        return math.isclose(self.value, other.value)

    def lowerbound(self) -> Self:
        return type(self)(-math.inf)

    def upperbound(self) -> Self:
        return type(self)(math.inf)


class AdditionCost(FloatingCost):
    def __add__(self, other: Self) -> Self:
        return type(self)(self.value + other.value)

    def identity(self) -> Self:
        return type(self)(0.0)


class MaximumCost(FloatingCost):
    def __add__(self, other: Self) -> Self:
        return type(self)(max(self.value, other.value))

    def identity(self) -> Self:
        return type(self)(-math.inf)


class MultiCost(Cost[tuple[FloatingCost, ...]]):
    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, index: int) -> FloatingCost:
        return self.value[index]

    def __add__(self, other: Self) -> Self:
        return type(self)(tuple(x + y for x, y in zip(self.value, other.value)))

    def isclose(self, other: Self) -> bool:
        return all(x.isclose(y) for x, y in zip(self.value, other.value))

    def lowerbound(self) -> Self:
        return type(self)(tuple(x.lowerbound() for x in self.value))

    def upperbound(self) -> Self:
        return type(self)(tuple(x.upperbound() for x in self.value))

    def identity(self) -> Self:
        return type(self)(tuple(x.identity() for x in self.value))


def make_knn_graph_with_dummy_target(
    X: pd.DataFrame,
    k: int,
    targets: list[int],
    cost_fn: Callable[[int, int], Cost],
    *,
    key: str,
) -> ig.Graph:
    adj = kneighbors_graph(X, k)
    graph: ig.Graph = ig.Graph.Adjacency(adj.toarray())  # type: ignore
    es = graph.es
    for e in es:
        u, v = e.tuple
        e[key] = cost_fn(u, v)
    min_cost = es[0][key].identity()
    t = graph.add_vertex().index
    graph.add_edges([(v, t) for v in targets], {key: [min_cost for _ in targets]})
    return graph


def multicost_shortest_paths(
    graph: ig.Graph,
    source: int,
    limit: int,
    *,
    key: str,
    verbose: bool,
) -> list[list[tuple[SupportsIndex, MultiCost]]]:
    dist_list = [[] for _ in range(graph.vcount())]
    min_cost = graph.es[0][key].identity()
    dist_list[source] = [(source, min_cost)]

    for _ in range(graph.vcount() - 1):
        for e in graph.es:
            u, v = e.tuple
            dist_u = dist_list[u]
            dist_v = dist_list[v]
            cost_uv = e[key]
            dist_list[v] = find_maxima_2d(
                dist_v + [(u, cost_su + cost_uv) for _, cost_su in dist_u],
                limit=limit,
                verbose=verbose,
            )
    return dist_list


def backtracking(
    graph: ig.Graph,
    dist_list: list[list[tuple[SupportsIndex, MultiCost]]],
    s: int,
    *,
    key: str,
    verbose: bool,
) -> list[list[int]]:
    t = graph.vcount() - 1
    paths = []
    for u, cost_sv in dist_list[t]:
        path = []
        v = t
        while s != v:
            dist_u = dist_list[u]
            path.append(u)

            eid = graph.get_eid(u, v)
            cost_uv = graph.es[eid][key]

            for i, cost_si in dist_u:
                if cost_sv.isclose(cost_si + cost_uv):
                    v = u
                    u = i
                    cost_sv = cost_si
                    break
            else:
                raise ValueError("No path found!")
        path.reverse()
        paths.append(path)

    if verbose:
        for i, ((_, cost), path) in enumerate(zip(dist_list[t], paths)):
            print(f"Path {i} {cost}: {path}")

    return paths


def find_maxima_2d(
    dist: list[tuple[SupportsIndex, MultiCost]],
    limit: int,
    *,
    verbose: bool,
) -> list[tuple[SupportsIndex, MultiCost]]:
    maxima = []
    if dist:
        dist.sort(key=itemgetter(1))
        _, y_min = dist[0][1].upperbound()
        for u, (x, y) in dist:
            if y < y_min:
                maxima.append((u, MultiCost((x, y))))
                y_min = y
        size = len(maxima)
        if size > limit:
            if verbose:
                warn(f"Exceed limit {size}!")
            step = (size - 1) / (limit - 1)
            maxima = [maxima[round(i * step)] for i in range(limit)]
    return maxima


def final_costs(dists: list[list[tuple[SupportsIndex, MultiCost]]]) -> list[MultiCost]:
    return [dist[1] for dist in dists[-1]]
