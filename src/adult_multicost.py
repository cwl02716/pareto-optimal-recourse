import math
from functools import partial

import fire
import pandas as pd
import seaborn as sns
import sklearn
from helper.adult import load_dataframe, select_rows_by_immutables
from helper.algorithm import (
    AdditionCost,
    MultiCost,
    backtracking,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.cmd import fire_cmd
from helper.common import get_sample
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

PATH = "dataset/50Ktrain.csv"

DROPS = ["fnlwgt", "education", "marital-status", "relationship", "occupation"]

IMMUTABLES = ["race", "sex", "native-country"]

YCOL = "50K"

sklearn.set_config(transform_output="pandas")
sns.set_style("white")
sns.set_context("paper")
# sns.set_palette("Spectral")


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

    # Remove outliers
    scaler = StandardScaler()
    X_std: pd.DataFrame = scaler.fit_transform(X_2d)  # type: ignore
    mask = (X_std.abs() <= 2).all(axis=1)
    X_2d_masked = X_2d[mask]
    y_masked = y[mask]

    sns.scatterplot(
        X_2d_masked,
        x=ft1,
        y=ft2,
        hue=y_masked,
        hue_norm=(0, 1),
        size=y_masked,
        sizes=(12, 18),
        size_order=(1, 0),
        ax=ax,
        alpha=0.6,
        palette="coolwarm",
    )
    ax.set_xlabel(ax.get_xlabel().upper())
    ax.set_ylabel(ax.get_ylabel().upper())
    plt.show()


def main(verbose: bool = True) -> None:
    def recourse_adult(
        samples: int = 256,
        neighbors: int = 4,
        limit: int = 8,
        *,
        seed: int = 0,
    ) -> None:
        X, y = select_rows_by_immutables(X_raw, y_raw, seed, IMMUTABLES)

        X, y = get_sample(
            X,
            y,
            samples,
            seed=seed,
            verbose=verbose,
            keep_first=True,
        )

        scaler = StandardScaler()
        X_scaled: pd.DataFrame = scaler.fit_transform(X)  # type: ignore

        ts = (y == 1).to_numpy().nonzero()[0].tolist()

        graph = make_knn_graph_with_dummy_target(
            X_scaled,
            neighbors,
            ts,
            partial(multi_costs_fn, X),
            key=key,
        )

        dists = multicost_shortest_paths(graph, 0, limit, key=key, verbose=verbose)

        paths = backtracking(
            graph,
            dists,
            0,
            samples,
            key=key,
            verbose=verbose,
        )

        component = PCA(n_components=2)
        component.fit(X_scaled, y)
        plot_images(X_scaled, y, paths, component)

    key = "cost"
    df_raw = load_dataframe(PATH, DROPS)
    X_raw = df_raw.drop(columns=YCOL)
    y_raw = df_raw[YCOL]
    fire_cmd(recourse_adult, "Adult-MultiCost")


if __name__ == "__main__":
    fire.Fire(main)
