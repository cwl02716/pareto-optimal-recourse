from os import PathLike

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler


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


def get_source_targets(
    y: pd.Series,
    source: int,
    target: int,
    *,
    show_n: int = 5,
    verbose: bool,
) -> tuple[int, list[int]]:
    s = (y.to_numpy() == source).nonzero()[0][0].item()
    ts = (y.to_numpy() == target).nonzero()[0].tolist()
    if verbose:
        print(f"Source: {s}, Targets: {ts[:show_n]} ...")
    return s, ts


def get_targets(y: pd.Series, target: int) -> list[int]:
    return (y.to_numpy() == target).nonzero()[0].tolist()
