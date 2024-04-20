import traceback
from os import PathLike
from typing import Any

import fire
import pandas as pd
from fire.core import FireExit
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler


def get_sample(
    X: pd.DataFrame, y: pd.Series, size: int, *, seed: int, verbose: bool
) -> tuple[pd.DataFrame, pd.Series]:
    X_sample = X.sample(size, random_state=seed)
    y_sample = y[X_sample.index]
    X_sample.reset_index(drop=True, inplace=True)
    y_sample.reset_index(drop=True, inplace=True)
    if verbose:
        print(f"Sampled dataset with size {size}")
    return X_sample, y_sample


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

    for ax, x in zip(axes, df.iloc[indices].to_numpy().reshape(n, 28, 28)):
        ax.imshow(x, cmap="gray")

    for ax in axes:
        ax.set_axis_off()

    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()
        if verbose:
            print(f"Saved image in {file}")


def load_dataframe(*, verbose: bool) -> tuple[pd.DataFrame, pd.Series]:
    if verbose:
        print("Starting fetching MNIST dataset...")
    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=True)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)  # type: ignore
    y = y.astype(int)  # type: ignore
    if verbose:
        print("Fetching MNIST dataset finished!")
    return X, y  # type: ignore


def fire_cmd(component: Any, name: str | None = None) -> None:
    prompt = "> "
    while True:
        try:
            fire.Fire(component, input(prompt), name)
        except FireExit:
            pass
        except EOFError:
            break
        except Exception:
            traceback.print_exc()
        finally:
            print()


def get_source_targets(
    X: pd.DataFrame,
    y: pd.Series,
    source: int,
    target: int,
    *,
    show_n: int = 5,
    verbose: bool = False,
) -> tuple[int, list[int]]:
    s = X[y == source].index[0]
    ts = X[y == target].index.tolist()
    if verbose:
        print(f"Source: {s}, Targets: {ts[:show_n]} ...")
    return s, ts
