import math
from functools import partial
from typing import Any

import pandas as pd
import sklearn
from algorithm import backtracking, make_knn_graph, recourse
from adult_helper import load_dataframe, select_rows_by_immutables
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

PATH = "dataset/50Ktrain.csv"

DROPS = ["fnlwgt", "education", "marital-status", "relationship", "occupation"]

IMMUTABLES = ["race", "sex", "native-country"]

YCOL = "50K"


def transform(
    df: pd.DataFrame,
    index: int,
    size: int,
    k: int,
    *,
    seed: Any = None,
) -> tuple[MinMaxScaler, pd.DataFrame, int]:
    X = df.drop(columns=YCOL)
    y = df[YCOL]

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled: pd.DataFrame = scaler.transform(X)  # type: ignore

    kmeans = KMeans(size, random_state=seed)
    kmeans.fit(X_scaled)
    X_small = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    knn = KNeighborsClassifier(k)
    knn.fit(X_scaled, y)

    y_small = pd.DataFrame(knn.predict(X_small), columns=[YCOL])
    df_small = pd.concat((X_small, y_small), axis=1)
    index_small = kmeans.predict(X_scaled.iloc[[index]]).item()

    return scaler, df_small, index_small


def cost_fn(df: pd.DataFrame, i: int, j: int) -> tuple[float, float]:
    time = 0.0
    payment = 0.0

    a = df.iloc[i]
    b = df.iloc[j]

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

    return time, payment


def show_path(
    df: pd.DataFrame,
    path: list[int],
    pca: PCA,
) -> None:
    ax: plt.Axes  # type: ignore
    fig, ax = plt.subplots()
    X = df.drop(columns=YCOL)
    y = df[YCOL]
    X_pca: pd.DataFrame = pca.transform(X)  # type: ignore
    X_path = X_pca.iloc[path]
    ax.plot("pca0", "pca1", "k:", data=X_path)
    ax.scatter("pca0", "pca1", c=y, data=X_pca)
    plt.show()


def main(
    index: int,
    samples: int,
    neighbors: int,
    limit: int,
    *,
    verbose: bool = False,
    seed: int = 0,
) -> None:
    df = load_dataframe(PATH, DROPS)

    df = select_rows_by_immutables(df, index, IMMUTABLES)

    scalar, df_small, s = transform(df, index, samples, neighbors, seed=seed)

    X = df_small.drop(columns=YCOL)
    y = df_small[YCOL]

    ts = (y == 1).to_numpy().nonzero()[0].tolist()

    graph = make_knn_graph(X, neighbors)

    dists = recourse(
        graph,
        s,
        ts,
        partial(cost_fn, df_small),
        limit,
        key="cost",
        verbose=verbose,
    )

    paths = backtracking(
        graph,
        dists,
        s,
        samples,
        key="cost",
        verbose=verbose,
    )

    pca = PCA(2)
    pca.fit(X)

    for path in paths:
        show_path(df_small, path, pca)


sklearn.set_config(transform_output="pandas")

if __name__ == "__main__":
    main(100, 256, 3, 10, verbose=True, seed=42)
