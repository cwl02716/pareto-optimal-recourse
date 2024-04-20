from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import fire
import igraph as ig
import pandas as pd
import sklearn
from algorithm import make_knn_graph_with_dummy_target
from mnist_helper import (
    fire_cmd,
    get_sample,
    get_source_targets,
    load_dataframe,
    plot_images,
)
from sklearn.neighbors import KernelDensity

sklearn.set_config(transform_output="pandas")


def kde_fn(df: pd.DataFrame, kde: KernelDensity, i: int, j: int) -> float:
    return kde.score_samples(df.iloc[[j]]).item()


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
        *,
        seed: int = 0,
    ) -> None:
        X_sample, y_sample = get_sample(X, y, samples, seed=seed, verbose=verbose)  # type: ignore

        s, ts = get_source_targets(X_sample, y_sample, source, target, verbose=verbose)
        graph = make_knn_graph_with_dummy_target(X_sample, neighbors)

        kde = KernelDensity()
        kde.fit(X_sample)

        paths = shortest_path(
            graph,
            s,
            ts,
            partial(kde_fn, X_sample, kde),
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
