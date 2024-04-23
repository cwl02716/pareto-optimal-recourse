from datetime import datetime
from functools import partial
from pathlib import Path
import random
from typing import Sequence

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

from helper.mnist import (
    load_dataframe,
    plot_images,
)

sklearn.set_config(transform_output="pandas")


def select_samples(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    *,
    seed: int = 0,
    startwith: Sequence[int] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    idx_set = set(y.index.tolist())
    idx_set.difference_update(startwith)
    rng = random.Random(seed)
    idx_list = list(startwith) + rng.sample(tuple(idx_set), n_samples - len(startwith))
    return X.loc[idx_list], y.loc[idx_list]


def get_targets(y: pd.Series, target: int) -> list[int]:
    return (y.to_numpy() == target).nonzero()[0].tolist()


def multi_costs_fn(X: pd.DataFrame, y: pd.Series, i: int, j: int) -> MultiCost:
    a = X.iloc[i].to_numpy()
    b = X.iloc[j].to_numpy()
    diff = np.subtract(a, b)
    cost_0 = abs(y.iat[i].item() - y.iat[j].item())
    cost_1 = np.linalg.norm(diff, 2).item()
    return MultiCost((MaximumCost(cost_0), AdditionCost(cost_1)))


def main(verbose: bool = True) -> None:
    def recourse_mnist(
        index: int,
        target: int = 8,
        n_samples: int = 256,
        k_neighbors: int = 4,
        limit: int = 8,
        *,
        seed: int = 0,
    ) -> None:
        X, y = select_samples(X_raw, y_raw, n_samples, seed=seed, startwith=(index,))
        s = 0
        ts = get_targets(y, target)

        source = y.iat[s].item()

        graph = make_knn_graph_with_dummy_target(
            X, k_neighbors, ts, partial(multi_costs_fn, X, y), key=key
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
            dir = Path(f"images/mnist_epsilon/{stamp}-{source}-{target}")
            dir.mkdir(exist_ok=True, parents=True)

            for i, path in enumerate(paths):
                plot_images(X, path, file=dir / f"{i}.png", verbose=verbose)

    key = "cost"
    X_raw, y_raw = load_dataframe(verbose=verbose)
    fire_cmd(recourse_mnist, "MNIST-Epsilon")


if __name__ == "__main__":
    fire.Fire(main)
