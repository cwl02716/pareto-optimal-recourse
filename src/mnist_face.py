from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import fire
import igraph as ig
import numpy as np
import pandas as pd
import sklearn
from algorithm import make_knn_graph
from mnist_helper import fire_cmd, get_sample, load_dataframe, plot_images

sklearn.set_config(transform_output="pandas")


def get_source_targets(
    X: pd.DataFrame, y: pd.Series, source: int, target: int, *, verbose: bool = False
) -> tuple[int, list[int]]:
    s = X[y == str(source)].index[0]
    ts = X[y == str(target)].index.tolist()
    if verbose:
        print(f"Source: {s}, Targets: {ts[:5]} ...")
    return s, ts


def l2_fn(df: pd.DataFrame, i: int, j: int) -> float:
    a = df.iloc[i]
    b = df.iloc[j]
    x = np.subtract(a, b)
    l2 = np.linalg.norm(x, 2, 0)
    return l2


def shortest_path(
    graph: ig.Graph,
    s: int,
    ts: list[int],
    cost_fn: Callable[[int, int], float],
    *,
    verbose: bool = False,
) -> list[list[int]]:
    for e in graph.es:
        u, v = e.tuple
        e["cost"] = cost_fn(u, v)
    assert all(w >= 0 for w in graph.es["cost"])
    return [
        x for x in graph.get_shortest_paths(s, ts, "cost", algorithm="dijkstra") if x
    ]


def main(verbose: bool = True) -> None:
    def face(
        source: int,
        target: int,
        samples: int = 256,
        neighbors: int = 4,
        limit: int = 8,
        *,
        seed: int = 0,
    ) -> None:
        X_sample, y_sample = get_sample(X, y, samples, seed=seed, verbose=verbose)  # type: ignore

        s, ts = get_source_targets(X_sample, y_sample, source, target, verbose=verbose)
        graph = make_knn_graph(X_sample, neighbors)

        paths = shortest_path(
            graph,
            s,
            ts,
            partial(l2_fn, X_sample),
            verbose=verbose,
        )

        dir = Path("images")
        dir.mkdir(exist_ok=True, parents=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        for i, path in enumerate(paths):
            name = f"mnist-{stamp}-{source}-{target}-{i}.png"
            plot_images(X_sample, path, file=dir / name, verbose=verbose)

    X, y = load_dataframe(verbose=verbose)

    fire_cmd(face, "MNIST-Face")


if __name__ == "__main__":
    fire.Fire(main)