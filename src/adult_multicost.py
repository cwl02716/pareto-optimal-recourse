from functools import partial
from warnings import warn

import pandas as pd
import seaborn as sns
import sklearn
import typer
from helper.adult import (
    get_targets,
    load_dataframe,
    non_outliers_mask,
    plot_images,
    select_actionable,
)
from helper.algorithm import (
    AdditionCost,
    MultiCost,
    backtracking,
    final_costs,
    find_maxima_nd,
    make_knn_graph_with_dummy_target,
    multicost_shortest_paths,
)
from helper.common import select_indices, select_mask, select_samples
from helper.shell import run_shell
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

sklearn.set_config(transform_output="pandas")
sns.set_style("white")
sns.set_context("talk")
sns.set_palette("bright")


def multi_costs_fn(X: pd.DataFrame, cols: list[str], i: int, j: int) -> MultiCost:
    a = X.iloc[i]
    b = X.iloc[j]
    return MultiCost(tuple(AdditionCost(abs(b.at[c] - a.at[c])) for c in cols))


def main(verbose: bool = True) -> None:
    def recourse_adult(
        index: int,
        source: int = 0,
        n_samples: int = 256,
        k_neighbors: int = 8,
        limit: int = 8,
        threshold: float = 0.75,
        *,
        seed: int = 0,
    ) -> None:
        X, y, X_std = select_actionable(
            X_raw, y_raw, X_raw_std, index=index, verbose=verbose
        )

        X, y, X_std = select_samples(
            X, y, X_std, n_samples=n_samples, seed=seed, verbose=verbose
        )

        model = MLPClassifier(learning_rate_init=0.01, max_iter=1024, random_state=seed)
        model.fit(X_std, y > 0.5)

        X, y, X_std = select_indices(
            X,
            y,
            X_std,
            indices=y.to_numpy().argsort(),
        )

        targets = get_targets(y.to_numpy(), threshold)

        graph = make_knn_graph_with_dummy_target(
            X_std,
            k_neighbors,
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
            method=find_maxima_nd,
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
            # title=f"Adult Dataset with Multi-cost ({", ".join(cols)})",
            n_scatter=128,
        )
        plt.show()

    key = "cost"
    cols = ["age", "education-num", "hours-per-week"]
    X_raw, y_raw = load_dataframe(verbose=verbose)

    scaler = StandardScaler()
    X_raw_std: pd.DataFrame = scaler.fit_transform(X_raw)  # type: ignore
    X_raw, y_raw, X_raw_std = select_mask(
        X_raw,
        y_raw,
        X_raw_std,
        mask=non_outliers_mask(X_raw_std, 3),
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_raw_std, y_raw)
    y_raw = pd.Series(
        model.predict_proba(X_raw_std)[:, 1],
        index=y_raw.index,
        name=y_raw.name,
    )

    run_shell(recourse_adult, "Adult-MultiCost")


if __name__ == "__main__":
    typer.run(main)
