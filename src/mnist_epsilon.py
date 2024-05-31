import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import product
from typing import Any, Sequence

import fire
import numpy as np
import pandas as pd
import sklearn

from helper.algorithm import (
    MaximumCost,
    MultiCost,
    final_costs,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.common import new_select_samples
from helper.mnist import get_targets, load_dataframe

sklearn.set_config(transform_output="pandas")


def multi_costs_fn(X: pd.DataFrame, y: pd.Series, i: int, j: int) -> MultiCost:
    a = X.iloc[i].to_numpy()
    b = X.iloc[j].to_numpy()
    diff = np.subtract(a, b)
    cost_0 = abs(y.iat[i].item() - y.iat[j].item())
    cost_1 = np.linalg.norm(diff, 2).item()
    return MultiCost((MaximumCost(cost_0), MaximumCost(cost_1)))


def recourse_mnist(
    X: pd.DataFrame,
    y: pd.Series,
    target: int,
    k_neighbor: int = 4,
    limit: int = 8,
    *,
    key: str = "cost",
) -> list[MultiCost]:
    ts = get_targets(y, target)
    graph = make_knn_graph_with_dummy_target(
        X,
        k_neighbor,
        ts,
        partial(multi_costs_fn, X, y),
        key=key,
    )
    dists = multicost_shortest_paths(graph, 0, limit, key=key, verbose=False)
    costs = final_costs(dists)
    return costs


def main(
    samples: Sequence[int],
    trials: int,
    index: int = 1,
    target: int = 8,
    seed: Any = None,
    verbose: bool = True,
) -> None:
    samples = sorted(samples, reverse=True)
    rng = random.Random(seed)
    X_raw, y_raw = new_select_samples(
        *load_dataframe(verbose=verbose),
        samples[0] * 2,
        rng=rng,
        startwith=(index,),
        verbose=verbose,
    )
    futs = []
    with ProcessPoolExecutor() as executor:
        for t in range(trials):
            X = X_raw
            y = y_raw
            for n in samples:
                if verbose:
                    print(f"submit trial: {t}, n_samples: {n}")
                X, y = new_select_samples(
                    X, y, n, rng=rng, startwith=(index,), verbose=verbose
                )
                fut = executor.submit(recourse_mnist, X, y, target)
                futs.append(fut)

    data = []
    for (n, t), future in zip(product(range(trials), samples), as_completed(futs)):
        if verbose:
            print(f"complete trial: {t}, n_samples: {n}")
        for i, costs in enumerate(future.result()):
            data.append((n, t, i, *map(float, costs)))

    df = pd.DataFrame(
        data,
        columns=("trial", "n_samples", "result", "cost_1", "cost_2"),
    )

    df.to_csv("dataset/mnist_epsilon.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
