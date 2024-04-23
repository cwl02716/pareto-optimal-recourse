from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from helper.common import select_mask

PATH = "dataset/50Ktrain.csv"
DROPS = ["fnlwgt", "education", "marital-status", "relationship"]
IMMUTABLES = ["sex", "race", "occupation", "native-country"]
YCOL = "50K"


class SupportsPredictProba(Protocol):
    def predict_proba(self, X: Any) -> NDArray: ...


def load_dataframe(
    path: str = PATH,
    drops: list[Any] = DROPS,
    *,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    if verbose:
        print("Starting reading Adult dataset...")
    df = pd.read_csv(Path(path))
    X = df.drop(columns=YCOL)
    y = df[YCOL]
    if drops:
        X = X.drop(columns=drops)
    if verbose:
        print("Reading Adult dataset finished!")
    return X, y


def non_outliers_mask(X: pd.DataFrame, threshold: float) -> NDArray:
    return (X.abs() <= threshold).all(1).to_numpy()


def proba_argsort(model: SupportsPredictProba, X: pd.DataFrame) -> NDArray:
    return model.predict_proba(X)[:, 1].argsort()


def get_targets(y: pd.Series, label: int) -> list[int]:
    return (y == label).to_numpy().nonzero()[0].tolist()


def select_actionable[*T](
    X: pd.DataFrame,
    *args: *T,
    index: int,
    verbose: bool,
    immutables: list[Any] = IMMUTABLES,
) -> tuple[pd.DataFrame, *T]:
    X_act = X.drop(columns=immutables)
    X_imm = X[immutables]
    mask = X_imm.eq(X_imm.iloc[index]).all(1).to_numpy()
    res = select_mask(X_act, *args, mask=mask)
    if verbose:
        print(f"Actionable dataset with size {res[0].shape[0]}")
    return res


def plot_images(
    X: pd.DataFrame,
    y: pd.Series,
    paths: list[list[int]],
    component: PCA,
    model: SupportsPredictProba,
) -> None:
    ax: plt.Axes  # type: ignore
    fig, ax = plt.subplots()
    X_2d: pd.DataFrame = component.transform(X)  # type: ignore
    ft1, ft2 = component.get_feature_names_out()
    sns.scatterplot(
        X_2d,
        x=ft1,
        y=ft2,
        hue=y,
        hue_norm=(0, 1),
        size=y,
        sizes=(6, 24),
        size_order=(1, 0),
        style=y,
        ax=ax,
        palette="coolwarm",
        alpha=0.9,
        ec="w",
        lw=0.2,
        zorder=len(paths) + 2,
    )

    bound = ax.axis()
    step = 256
    j = complex(0, step)
    grid = np.mgrid[bound[0] : bound[1] : j, bound[2] : bound[3] : j]
    X_grid = pd.DataFrame(
        component.inverse_transform(grid.reshape(2, step * step).T),
        columns=component.feature_names_in_,
    )
    y_grid = model.predict_proba(X_grid)[:, 1].reshape(step, step)
    ct = ax.contourf(
        grid[0],
        grid[1],
        y_grid,
        levels=5,
        vmin=0,
        vmax=1,
        alpha=1 / 3,
        zorder=0,
        cmap="coolwarm",
    )
    cb = fig.colorbar(
        ct,
        ax=ax,
        label="Probability",
    )
    cb.solids.set_alpha(1)  # type: ignore

    for i, path in enumerate(paths):
        X_path = X_2d.iloc[path]
        sns.lineplot(
            X_path,
            x=ft1,
            y=ft2,
            sort=False,
            ax=ax,
            label=f"Path {i}",
            lw=2,
            path_effects=[
                patheffects.SimpleLineShadow((0.5, -0.5), "k", 0.5),
                patheffects.Normal(),
            ],
            alpha=0.8,
        )

    ax.set_xlabel(ax.get_xlabel().upper())
    ax.set_ylabel(ax.get_ylabel().upper())
