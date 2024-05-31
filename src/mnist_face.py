from datetime import datetime
from functools import partial
from pathlib import Path

import fire
import igraph as ig
import pandas as pd
import sklearn
from helper.algorithm import AdditionCost, make_knn_graph_with_dummy_target
from helper.cmd import fire_cmd
from helper.common import select_samples
from helper.mnist import (
    get_source_targets,
    load_dataframe,
    plot_images,
)
from sklearn.neighbors import KernelDensity

sklearn.set_config(transform_output="pandas")


def kde_fn(df: pd.DataFrame, kde: KernelDensity, i: int, j: int) -> AdditionCost:
    return AdditionCost(-kde.score_samples(df.iloc[[j]]).item())


def shortest_path(graph: ig.Graph, s: int, t: int, *, key: str) -> list[int]:
    paths = graph.get_shortest_paths(s, t, key, algorithm="bellman_ford")
    assert len(paths) == 1
    return paths[0][:-1]


def main(verbose: bool = True) -> None:
    def face(
        source: int,
        target: int,
        samples: int = 256,
        neighbors: int = 4,
        *,
        seed: int = 0,
    ) -> None:
        X, y = select_samples(
            X_raw, y_raw, n_samples=samples, seed=seed, verbose=verbose
        )

        s, ts = get_source_targets(X, y, source, target, verbose=verbose)

        kde = KernelDensity()
        kde.fit(X)

        graph = make_knn_graph_with_dummy_target(
            X,
            neighbors,
            ts,
            partial(kde_fn, X, kde),
            key=key,
        )

        if verbose:
            print("Starting Bellman-Ford algorithm...")

        path = shortest_path(graph, s, samples, key=key)

        if verbose:
            print("Bellman-Ford algorithm finished!")

        if path:
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            dir = Path("images/mnist_face")
            dir.mkdir(exist_ok=True, parents=True)
            plot_images(
                X,
                path,
                file=dir / f"{stamp}-{source}-{target}.png",
                verbose=verbose,
            )

    key = "cost"
    X_raw, y_raw = load_dataframe(verbose=verbose)
    fire_cmd(face, "MNIST-Face")


if __name__ == "__main__":
    fire.Fire(main)
