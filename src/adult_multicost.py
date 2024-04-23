import math
from functools import partial
from typing import SupportsIndex
from warnings import warn

import fire
import pandas as pd
import seaborn as sns
import sklearn
from helper.adult import (
    get_targets,
    load_dataframe,
    non_outliers_mask,
    plot_images,
    proba_argsort,
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
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

sklearn.set_config(transform_output="pandas")
sns.set_style("white")
sns.set_context("paper")
sns.set_palette("bright")


def multi_costs_fn(X: pd.DataFrame, cols: tuple[str, str], i: int, j: int) -> MultiCost:
    def cost_fn(a: pd.Series, b: pd.Series, key: str) -> float:
        x = b.at[key] - a.at[key]
        return x if x > 0 else math.inf

    a = X.iloc[i]
    b = X.iloc[j]
    cost_age = cost_fn(a, b, cols[0])
    cost_education = cost_fn(a, b, cols[1])
    return MultiCost((AdditionCost(cost_age), AdditionCost(cost_education)))


def final_costs(dists: list[list[tuple[SupportsIndex, MultiCost]]]) -> list[MultiCost]:
    return [dist[1] for dist in dists[-1]]


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

        model = MLPClassifier((4, 4), learning_rate_init=0.03, random_state=seed)
        model.fit(X_std, y)
        X, y, X_std = select_indices(X, y, X_std, indices=proba_argsort(model, X_std))

        targets = get_targets(y, 1)

        graph = make_knn_graph_with_dummy_target(
            X_std,
            neighbors,
            targets,
            partial(multi_costs_fn, X, cols),
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

        pca = PCA(n_components=2)
        pca.fit(X_std)

        if not paths:
            warn("No paths found!")

        for path in paths:
            print(X_raw.loc[X.index[path]])

        costs = final_costs(dists)
        plot_images(
            X_std,
            y,
            paths,
            costs,
            pca,
            model,
            title=f"Adult Dataset with Multi-cost ({", ".join(cols)})",
            samples=64,
        )
        plt.show()

    key = "cost"
    cols = "age", "hours-per-week"
    X_raw, y_raw = load_dataframe(verbose=verbose)

    scaler = StandardScaler()
    X_raw_std: pd.DataFrame = scaler.fit_transform(X_raw)  # type: ignore
    X_raw, y_raw, X_raw_std = select_mask(
        X_raw,
        y_raw,
        X_raw_std,
        mask=non_outliers_mask(X_raw_std, 3.0),
    )

    model = RandomForestClassifier(random_state=42)
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
