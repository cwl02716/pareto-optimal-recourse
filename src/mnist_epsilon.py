import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import product
from typing import Any, Sequence

import fire
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import radius_neighbors_graph
from helper.algorithm import (
    AdditionCost,
    MaximumCost,
    MultiCost,
    final_costs,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.mnist import load_dataframe

sklearn.set_config(transform_output="pandas")


def select_samples(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    *,
    rng: random.Random,
    startwith: Sequence[int] = (),
) -> tuple[pd.DataFrame, pd.Series]:
    idx_set = set(y.index.tolist())
    idx_set.difference_update(startwith)
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


def recourse_mnist(
    X: pd.DataFrame,
    y: pd.Series,
    target: int,
    k_neighbor: int = 4,
    limit: int = 8,
    *,
    key: str = "cost",
) -> list[tuple[float, ...]]:
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
    rng = random.Random(seed)
    index = 1
    target = 8

    X_raw, y_raw = load_dataframe(verbose=verbose)

    futs = []
    with ProcessPoolExecutor() as executor:
        for n, t in product(samples, range(trials)):
            if verbose:
                print(f"submit n_samples: {n}, trial: {t}")
            X, y = select_samples(
                X_raw,
                y_raw,
                n,
                rng=rng,
                startwith=(index,),
            )
            fut = executor.submit(recourse_mnist, X, y, target)
            futs.append(fut)

    data = []
    for (n, t), future in zip(product(samples, range(trials)), as_completed(futs)):
        if verbose:
            print(f"complete n_samples: {n}, trial: {t}")
        for i, costs in enumerate(future.result()):
            data.append((n, t, i, *costs))

    df = pd.DataFrame(
        data,
        columns=("n_samples", "trial", "result", "cost_1", "cost_2"),
    )

    df.to_csv("dataset/mnist_epsilon.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
