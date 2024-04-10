import traceback
from datetime import datetime
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
import sklearn
from algorithm import backtracking, recourse
from fire.core import FireExit
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

sklearn.set_config(transform_output="pandas")


def get_sample(
    X: pd.DataFrame, y: pd.Series, size: int, *, seed: Any = None, verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    X_sample = X.sample(size, random_state=seed)
    y_sample = y[X_sample.index]
    X_sample.reset_index(drop=True, inplace=True)
    y_sample.reset_index(drop=True, inplace=True)
    if verbose:
        print(f"Sampled dataset with size {size}")
    return X_sample, y_sample


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


def plot_images(
    df, indices: list[int], *, file: PathLike | None = None, verbose: bool = False
) -> None:
    fig, axes = plt.subplots(
        1,
        len(indices),
        layout="tight",
        squeeze=False,
        subplot_kw={"xticks": [], "yticks": []},
    )
    axes = axes[0]
    for ax, i in zip(axes, indices):
        ax.imshow(df.iloc[i].to_numpy().reshape(28, 28), cmap="gray")
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()
        if verbose:
            print(f"Saved image in {file}")


def main(verbose: bool = False) -> None:
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
        graph, dists = recourse(
            X_sample,
            neighbors,
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

    if verbose:
        print("Starting fetching MNIST dataset...")
    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=True)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)  # type: ignore
    if verbose:
        print("Fetching MNIST dataset finished!")

    prompt = "> "
    while True:
        try:
            fire.Fire(recourse_mnist, input(prompt))

        except FireExit:
            pass

        except EOFError:
            break

        except Exception:
            traceback.print_exc()

        finally:
            print()


if __name__ == "__main__":
    fire.Fire(main)
