import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import typer
from helper.algorithm import (
    AdditionCost,
    MaximumCost,
    MultiCosts,
    backtracking,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.common import new_select_samples
from helper.mnist import (
    get_targets,
    load_dataframe,
    plot_images,
)
from helper.shell import run_shell

sklearn.set_config(transform_output="pandas")


def multi_costs_fn(X: pd.DataFrame, y: pd.Series, i: int, j: int) -> MultiCosts:
    a = X.iloc[i]
    b = X.iloc[j]
    l1 = abs(y.iat[j].item() - y.iat[i].item())
    l2 = np.linalg.norm(a - b, 2).item()
    return MultiCosts((MaximumCost(l1), AdditionCost(l2)))


def main(verbose: bool = True) -> None:
    def recourse_mnist(
        index: int,
        target: int,
        n_samples: int = 256,
        k_neighbors: int = 4,
        limit: int = 8,
        *,
        seed: Any = None,
    ) -> None:
        rng = random.Random(seed)
        X, y = new_select_samples(
            X_raw,
            y_raw,
            n_samples,
            rng=rng,
            startwith=(index,),
            verbose=verbose,
        )

        s = 0
        source = y.iat[s].item()
        ts = get_targets(y, target)

        graph = make_knn_graph_with_dummy_target(
            X,
            k_neighbors,
            ts,
            partial(multi_costs_fn, X, y),
            key=key,
        )

        if verbose:
            print("Starting recourse algorithm...")

        dists = multicost_shortest_paths(graph, 0, limit, key=key, verbose=verbose)

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
                plot_images(X, path, file=dir / f"{i}.pdf", verbose=verbose)

    key = "cost"
    X_raw, y_raw = load_dataframe(verbose=verbose)
    run_shell(recourse_mnist, "MNIST-MultiCost")


if __name__ == "__main__":
    typer.run(main)
