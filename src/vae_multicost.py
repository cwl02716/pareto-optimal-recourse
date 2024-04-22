from datetime import datetime
from functools import partial
from os import PathLike
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import sklearn
from helper.algorithm import (
    MaximumCost,
    MultiCost,
    backtracking,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.cmd import fire_cmd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

sklearn.set_config(transform_output="pandas")


def load_dataframe() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    path = Path("dataset/vae.csv.gz")
    X = pd.read_csv(path)
    y = X["epoch"]
    nll = X["nll"]
    X = X.drop(columns=["nll", "epoch"])
    return X, y, nll


def multi_costs_fn(X, y, nll, i, j) -> MultiCost:
    Xi = X.iloc[i]
    Xj = X.iloc[j]
    cost_1 = np.square(Xi - Xj).mean()
    cost_2 = nll.iat[j]
    return MultiCost((MaximumCost(cost_1), MaximumCost(cost_2)))


def get_source_targets(
    y: pd.Series,
    source: int,
    target: int,
    *,
    show_n: int = 5,
    seed: int,
    verbose: bool,
) -> tuple[int, list[int]]:
    rng = np.random.default_rng(seed)
    arr = y.to_numpy()
    # intentionally swapped source and targets
    s = rng.choice(np.nonzero(arr == target)[0], 1).item()
    ts = np.nonzero(arr == source)[0].tolist()
    if verbose:
        print(f"Source: {s}, Targets: {ts[:show_n]} ...")
    return s, ts


def plot_images(
    df: pd.DataFrame,
    indices: list[int],
    *,
    file: PathLike[str] | None,
    verbose: bool,
) -> None:
    n = len(indices)
    cols = 5
    fig, _axes = plt.subplots(
        n // cols + (n % cols > 0),
        cols,
        layout="tight",
        subplot_kw={"xticks": [], "yticks": []},
    )
    axes: list[Axes] = _axes.ravel().tolist()
    arr = df.iloc[indices].to_numpy().clip(0, 1)
    arr = arr.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)
    for ax, x in zip(axes, arr):
        ax.imshow(x, vmin=0, vmax=1)
    for ax in axes:
        ax.set_axis_off()
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()
        if verbose:
            print(f"Saved image in {file}")


def main(verbose: bool = True) -> None:
    def vae_mnist(
        source: int,
        target: int,
        neighbors: int = 4,
        limit: int = 8,
        *,
        seed: int = 0,
    ) -> None:
        s, ts = get_source_targets(y, source, target, seed=seed, verbose=verbose)

        graph = make_knn_graph_with_dummy_target(
            X,
            neighbors,
            ts,
            partial(multi_costs_fn, X, y, nll),
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
            dir = Path(f"images/vae_multicost/{stamp}-{source}-{target}")
            dir.mkdir(exist_ok=True, parents=True)

            for i, path in enumerate(paths):
                plot_images(X, path, file=dir / f"{i}.png", verbose=verbose)

        else:
            print("Recourse not found!")

    key = "cost"
    X, y, nll = load_dataframe()
    fire_cmd(vae_mnist, "VAE-MultiCost")


if __name__ == "__main__":
    fire.Fire(main)
