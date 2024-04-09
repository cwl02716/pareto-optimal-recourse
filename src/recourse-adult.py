from typing import Any

import pandas as pd
import sklearn
from algo import backtracking, recourse
from helper import load_dataframe, select_rows_by_immutables
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


def main(index: int, size: int, k: int, limit: int, *, seed: int) -> None:
    df = load_dataframe(PATH, DROPS)

    df = select_rows_by_immutables(df, index, IMMUTABLES)

    scalar, df_small, s = transform(df, index, size, k, seed=seed)

    X = df_small.drop(columns=YCOL)
    y = df_small[YCOL]
    graph, ts, dists = recourse(X, y, k, s, limit=limit, verbose=False)

    paths = backtracking(graph, dists, s, size)

    pca = PCA(2)
    pca.fit(X)

    for path in paths:
        print(path)
        show_path(df_small, path, pca)
        break
    else:
        print("No path found")


sklearn.set_config(transform_output="pandas")

if __name__ == "__main__":
    main(0, 256, 3, 10, seed=42)
