from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import total_ordering
from operator import add, itemgetter
from typing import Any, Protocol, Self, SupportsIndex
from warnings import warn

import igraph as ig
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

logger = logging.getLogger(__name__)


@total_ordering
class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, other: object, /) -> bool: ...

    @abstractmethod
    def __lt__(self, other: Self, /) -> bool: ...


class Cost(Comparable):
    @abstractmethod
    def __add__(self, other: Self) -> Self: ...

    @abstractmethod
    def lowerbound(self) -> Self: ...

    @abstractmethod
    def upperbound(self) -> Self: ...

    @abstractmethod
    def identity(self) -> Self: ...


@dataclass(order=True, slots=True)
class FloatingCost(Cost):
    value: float

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value:.2g}"

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


@dataclass(init=False, order=True, slots=True)
class MultiCosts(Cost):
    value: Sequence[FloatingCost]

    def __init__(self, value: Iterable[FloatingCost]) -> None:
        self.value = (*value,)

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, index: int) -> FloatingCost:
        return self.value[index]

    def __iter__(self) -> Iterator[FloatingCost]:
        return iter(self.value)

    def __add__(self, other: Self) -> Self:
        return type(self)(map(add, self.value, other.value))

    def __repr__(self) -> str:
        return repr(self.value)

    def isclose(self, other: Self) -> bool:
        return all(map(math.isclose, self.value, other.value))

    def lowerbound(self) -> Self:
        return type(self)(x.lowerbound() for x in self.value)

    def upperbound(self) -> Self:
        return type(self)(x.upperbound() for x in self.value)

    def identity(self) -> Self:
        return type(self)(x.identity() for x in self.value)


# intended to replace `make_knn_graph_with_dummy_target`
def make_graph[T](
    X: pd.DataFrame,
    targets: Sequence[int],
    k: T,
    *,
    key: str,
    cost_fn: Callable[[int, int], Cost],
    maker_fn: Callable[[pd.DataFrame, T], sparse.spmatrix] = kneighbors_graph,
) -> ig.Graph:
    adj = maker_fn(X, k).astype(np.int32)  # type: ignore
    g: ig.Graph = ig.Graph.Adjacency(adj)
    logger.debug("G(V=%d, E=%d) | build from adj matrix", g.vcount(), g.ecount())

    idx = X.index.to_list()
    g.es[key] = costs = [cost_fn(idx[u], idx[v]) for u, v in g.get_edgelist()]
    assert costs, "empty costs"

    t = g.add_vertex()
    i = costs[0].identity()
    g.add_edges([(v, t) for v in targets], {key: [i] * len(targets)})
    logger.debug("G(V=%d, E=%d) | add target vertex", g.vcount(), g.ecount())

    return g


def make_knn_graph_with_dummy_target(
    X: pd.DataFrame,
    k: int | float,
    targets: list[int],
    cost_fn: Callable[[int, int], Cost],
    *,
    key: str,
    func: Callable[..., Any] = kneighbors_graph,
) -> ig.Graph:
    adj = func(X, k)
    graph: ig.Graph = ig.Graph.Adjacency(adj)  # type: ignore
    es = graph.es
    for e in es:
        u, v = e.tuple
        e[key] = cost_fn(u, v)
    min_cost = es[0][key].identity()
    t = graph.add_vertex().index
    graph.add_edges([(v, t) for v in targets], {key: [min_cost for _ in targets]})
    return graph


def prune[T](x: list[T], limit: int, *, verbose: bool) -> list[T]:
    size = len(x)
    if size > limit:
        if verbose:
            warn(f"Exceed limit {size}!")
        step = (size - 1) / (limit - 1)
        x = [x[round(i * step)] for i in range(limit)]
    return x


def find_maxima_2d(
    dist: list[tuple[SupportsIndex, MultiCosts]],
    limit: int,
    *,
    verbose: bool,
) -> list[tuple[SupportsIndex, MultiCosts]]:
    maxima = []
    if dist:
        dist.sort(key=itemgetter(1))
        _, y_min = dist[0][1].upperbound()
        for u, (x, y) in dist:
            if y < y_min:
                maxima.append((u, MultiCosts((x, y))))
                y_min = y
        maxima = prune(maxima, limit, verbose=verbose)
    return maxima


def find_maxima_nd(
    dist: list[tuple[SupportsIndex, MultiCosts]],
    limit: int,
    *,
    verbose: bool,
) -> list[tuple[SupportsIndex, MultiCosts]]:
    maxima = []
    if dist:
        tmp = []
        for pu, cu in dist:
            if not any(cu == cv for _, cv in tmp):
                tmp.append((pu, cu))
        for i, (pu, cu) in enumerate(tmp):
            if not any(
                all(cui >= cvi for cui, cvi in zip(cu, cv))
                for j, (pv, cv) in enumerate(tmp)
                if (i != j)
            ):
                maxima.append((pu, cu))
        maxima = prune(maxima, limit, verbose=verbose)
    return maxima


def multicost_shortest_paths(
    graph: ig.Graph,
    source: int,
    limit: int,
    *,
    key: str,
    verbose: bool,
    method: Callable[..., Any] = find_maxima_2d,
) -> list[list[tuple[SupportsIndex, MultiCosts]]]:
    dist_list = [[] for _ in range(graph.vcount())]
    min_cost = graph.es[0][key].identity()
    dist_list[source] = [(source, min_cost)]

    for _ in range(graph.vcount() - 1):
        for e in graph.es:
            u, v = e.tuple
            dist_u = dist_list[u]
            dist_v = dist_list[v]
            cost_uv = e[key]
            dist_list[v] = method(
                dist_v + [(u, cost_su + cost_uv) for _, cost_su in dist_u],
                limit=limit,
                verbose=verbose,
            )
    return dist_list


def backtracking(
    graph: ig.Graph,
    dist_list: list[list[tuple[SupportsIndex, MultiCosts]]],
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


def final_costs(
    dists: list[list[tuple[SupportsIndex, MultiCosts]]],
) -> list[MultiCosts]:
    return [dist[1] for dist in dists[-1]]
