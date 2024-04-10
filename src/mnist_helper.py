import traceback
from os import PathLike
from typing import Any

import fire
import pandas as pd
from fire.core import FireExit
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler


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


def load_dataframe(
    *, scale: bool = True, verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    if verbose:
        print("Starting fetching MNIST dataset...")
    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=True)
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)  # type: ignore
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
