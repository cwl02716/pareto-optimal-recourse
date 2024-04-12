from datetime import datetime
from functools import partial
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import sklearn
from algorithm import backtracking, make_knn_graph, recourse
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


def multi_costs_fn(df: pd.DataFrame, i: int, j: int) -> list[tuple[float, float]]:
    a = df.iloc[i]
    b = df.iloc[j]

    x = np.subtract(a, b)
    l2 = np.linalg.norm(x, 2, 0)

    x = np.abs(x, x)
    sum_of_diff = x.sum()

    x = np.maximum.reduce((a, b), out=x)
    sum_of_max = x.sum()

    l1 = sum_of_diff / sum_of_max

    return [(l1, l2)]


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
        X_sample, y_sample = get_sample(X, y, samples, seed=seed, verbose=verbose)  # type: ignore
        s, ts = get_source_targets(X_sample, y_sample, source, target, verbose=verbose)
        graph = make_knn_graph(X_sample, neighbors)
        dists = recourse(
            graph,
            s,
            ts,
            partial(multi_costs_fn, X_sample),
            limit=limit,
            verbose=verbose,
        )
        paths = backtracking(graph, dists, s, samples, verbose=verbose)

        dir = Path("images")
        dir.mkdir(exist_ok=True, parents=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        for i, path in enumerate(paths):
            name = f"mnist-{stamp}-{source}-{target}-{i}.png"
            plot_images(X_sample, path, file=dir / name, verbose=verbose)

    X, y = load_dataframe(verbose=verbose)

    fire_cmd(recourse_mnist, "MNIST-MultiCost")


if __name__ == "__main__":
    fire.Fire(main)
