from calendar import c
import math
from functools import partial
from warnings import warn

import fire
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from helper.adult import (
    load_dataframe,
    select_actionable,
)
from helper.algorithm import (
    AdditionCost,
    MultiCost,
    backtracking,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.cmd import fire_cmd
from helper.common import select_indices, select_mask, select_samples
from matplotlib import colors, patheffects, pyplot as plt, ticker
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression

sklearn.set_config(transform_output="pandas")
sns.set_style("white")
sns.set_context("paper")
sns.set_palette("bright")


def non_outliers_mask(X: pd.DataFrame, threshold: float) -> NDArray:
    return (X.abs() <= threshold).all(1).to_numpy()


def proba_argsort(model: LogisticRegression, X: pd.DataFrame) -> NDArray:
    return model.predict_proba(X)[:, 1].argsort()


def get_targets(y: pd.Series, label: int) -> list[int]:
    return (y == label).to_numpy().nonzero()[0].tolist()


def multi_costs_fn(X: pd.DataFrame, i: int, j: int) -> MultiCost:
    time = 0.0
    payment = 0.0

    a = X.iloc[i]
    b = X.iloc[j]

    # for age
    time = max(time, b["age"] - a["age"])

    # education
    time = max(time, b["education-num"] - a["education-num"])

    # workclass
    time = max(time, abs(b["workclass"] - a["workclass"]))

    time = max(
        time,
        (b["capital-gain"] ** 2)
        + (b["capital-loss"] ** 2)
        - (a["capital-gain"] ** 2)
        - (a["capital-loss"] ** 2),
    )

    # sigmoid(workclass : hours-per-week)
    eps = 1e-3
    m = a["workclass"] / (a["hours-per-week"] + eps)
    m -= b["workclass"] / (b["hours-per-week"] + eps)
    payment += 1.0 / (1.0 + 1.44 * math.exp(m))  # add bias

    return MultiCost((AdditionCost(time), AdditionCost(payment)))


def plot_images(
    X: pd.DataFrame,
    y: pd.Series,
    paths: list[list[int]],
    component: PCA,
    model: LogisticRegression,
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
        levels=10,
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
                # patheffects.Stroke(linewidth=3, foreground='w'),
                patheffects.Normal(),
            ],
            alpha=0.8,
        )

    ax.set_xlabel(ax.get_xlabel().upper())
    ax.set_ylabel(ax.get_ylabel().upper())
    plt.show()


def main(verbose: bool = True) -> None:
    def recourse_adult(
        index: int,
        source: int = 0,
        samples: int = 256,
        neighbors: int = 4,
        limit: int = 8,
        *,
        seed: int = 0,
    ) -> None:
        X, y, X_std = select_actionable(
            X_raw, y_raw, X_raw_std, index=index, verbose=verbose
        )

        X, y, X_std = select_samples(
            X, y, X_std, samples=samples, seed=seed, verbose=verbose
        )

        model = LogisticRegression(random_state=seed)
        model.fit(X_std, y)
        X, y, X_std = select_indices(X, y, X_std, indices=proba_argsort(model, X_std))

        targets = get_targets(y, 1)

        graph = make_knn_graph_with_dummy_target(
            X_std,
            neighbors,
            targets,
            partial(multi_costs_fn, X),
            key=key,
        )

        dists = multicost_shortest_paths(
            graph,
            source,
            limit,
            key=key,
            verbose=verbose,
        )

        paths = backtracking(
            graph,
            dists,
            source,
            key=key,
            verbose=verbose,
        )

        if paths:
            for path in paths:
                print(X_raw.loc[X.index[path]])

            pca = PCA(n_components=2)
            pca.fit(X_std)
            plot_images(X_std, y, paths, pca, model)

        else:
            warn("No paths found!")

    key = "cost"
    X_raw, y_raw = load_dataframe(verbose=verbose)

    scaler = StandardScaler()
    X_raw_std: pd.DataFrame = scaler.fit_transform(X_raw)  # type: ignore
    X_raw, y_raw, X_raw_std = select_mask(
        X_raw,
        y_raw,
        X_raw_std,
        mask=non_outliers_mask(X_raw_std, 3),
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_raw_std, y_raw)
    X_raw, y_raw, X_raw_std = select_indices(
        X_raw,
        y_raw,
        X_raw_std,
        indices=proba_argsort(model, X_raw_std),
    )

    fire_cmd(recourse_adult, "Adult-MultiCost")


if __name__ == "__main__":
    fire.Fire(main)
