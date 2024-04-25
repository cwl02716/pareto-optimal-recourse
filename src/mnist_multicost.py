from datetime import datetime
from functools import partial
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import sklearn
from helper.algorithm import (
    AdditionCost,
    MaximumCost,
    MultiCost,
    backtracking,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.cmd import fire_cmd
from helper.common import select_samples
from helper.mnist import (
    get_source_targets,
    load_dataframe,
    plot_images,
)

sklearn.set_config(transform_output="pandas")


def multi_costs_fn(X: pd.DataFrame, y: pd.Series, i: int, j: int) -> MultiCost:
    a = X.iloc[i]
    b = X.iloc[j]
    l1 = abs(y.iat[j].item() - y.iat[i].item())
    l2 = np.linalg.norm(a - b, 2).item()
    return MultiCost((MaximumCost(l1), AdditionCost(l2)))


def main(verbose: bool = True) -> None:
    def recourse_mnist(
        source: int,
        target: int,
        samples: int = 256,
        neighbors: int = 4,
        limit: int = 8,
        *,
        seed: int = 0,
    ) -> None:
        X, y = select_samples(X_raw, y_raw, samples=samples, seed=seed, verbose=verbose)

        s, ts = get_source_targets(X, y, source, target, verbose=verbose)

        graph = make_knn_graph_with_dummy_target(
            X,
            neighbors,
            ts,
            partial(multi_costs_fn, X, y),
            key=key,
        )

        if verbose:
            print("Starting recourse algorithm...")

        dists = multicost_shortest_paths(graph, s, limit, key=key, verbose=verbose)

        if verbose:
            print("Recourse algorithm finished!")

        paths = backtracking(
            graph,
            dists,
            s,
            key=key,
            verbose=verbose,
        )

        if paths:
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            dir = Path(f"images/mnist_multicost/{stamp}-{source}-{target}")
            dir.mkdir(exist_ok=True, parents=True)

            for i, path in enumerate(paths):
                plot_images(X, path, file=dir / f"{i}.png", verbose=verbose)

    key = "cost"
    X_raw, y_raw = load_dataframe(verbose=verbose)
    fire_cmd(recourse_mnist, "MNIST-MultiCost")


if __name__ == "__main__":
    fire.Fire(main)
