import math
from functools import partial
from warnings import warn

import fire
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
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
from helper.common import select_by_indices, select_by_mask, select_samples
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

sklearn.set_config(transform_output="pandas")
sns.set_style("white")
sns.set_context("paper")


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
        sizes=(8, 24),
        size_order=(1, 0),
        ax=ax,
        palette="coolwarm",
        alpha=0.8,
        zorder=len(paths) + 8,
    )
    for i, (path, color) in enumerate(zip(paths, sns.color_palette("bright"))):
        X_path = X_2d.iloc[path]
        sns.lineplot(
            X_path,
            x=ft1,
            y=ft2,
            sort=False,
            ax=ax,
            label=f"Path {i}",
            color=color,
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
            X_raw,
            y_raw,
            X_raw_std,
            index=index,
        )

        X, y, X_std = select_samples(
            X,
            y,
            X_std,
            samples=samples,
            seed=seed,
            verbose=verbose,
        )

        model = LogisticRegression(random_state=seed)
        model.fit(X_std, y)
        X, y, X_std = select_by_indices(
            X,
            y,
            X_std,
            indices=model.predict_proba(X_std)[:, 1].argsort(),
        )

        targets = (y == 1).to_numpy().nonzero()[0].tolist()

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
            graph.vcount() - 1,
            key=key,
            verbose=verbose,
        )

        if paths:
            for path in paths:
                print(X_raw.loc[X.index[path]])

            pca = PCA(n_components=2)
            pca.fit(X_std)
            plot_images(X_std, y, paths, pca)
        else:
            warn("No paths found!")

    key = "cost"
    X_raw, y_raw = load_dataframe(verbose=verbose)

    scaler = StandardScaler()
    X_raw_std: pd.DataFrame = scaler.fit_transform(X_raw)  # type: ignore
    X_raw, y_raw, X_raw_std = select_by_mask(
        X_raw,
        y_raw,
        X_raw_std,
        mask=(X_raw_std.abs() <= 3).all(1).to_numpy(),
    )

    model = LogisticRegression(random_state=0)
    model.fit(X_raw_std, y_raw)
    X_raw, y_raw, X_raw_std = select_by_indices(
        X_raw,
        y_raw,
        X_raw_std,
        indices=model.predict_proba(X_raw_std)[:, 1].argsort(),
    )

    fire_cmd(recourse_adult, "Adult-MultiCost")


if __name__ == "__main__":
    fire.Fire(main)
