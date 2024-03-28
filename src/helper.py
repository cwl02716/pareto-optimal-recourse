from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

__all__ = ["load_dataframe", "transfrom_dataframe", "YCOL"]

YCOL = "50K"


def _select_same(
    df: pd.DataFrame,
    index: int,
    immutables: list[Any] = [
        "race",
        "sex",
        "native-country",
    ],
) -> pd.DataFrame:
    X = df[immutables]
    i = X.eq(X.iloc[index]).all(1)
    return df[i].drop(columns=immutables)


def _fit_minmax(X: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler


def _fit_kmeans(
    X: pd.DataFrame,
    k: int,
    seed: Any = None,
) -> KMeans:
    kmeans = KMeans(k, random_state=seed)
    kmeans.fit(X)
    return kmeans


def _fit_knn(X: pd.DataFrame, y: pd.Series, k: int) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(k)
    knn.fit(X, y)
    return knn


def _fit_pca(X: pd.DataFrame, n_components: int = 2) -> PCA:
    pca = PCA(n_components)
    pca.fit(X)
    return pca


def show_path(df: pd.DataFrame, path: list[int], pca: PCA):
    X = df.drop(columns=YCOL)
    X_path = X.iloc[path]
    X_pca = pca.transform(X_path)
    plt.plot(X_pca[:, 0], X_pca[:, 1], "o-")
    plt.show()


def load_dataframe(
    path: str = "dataset/50Ktrain.csv",
    drops: list[Any] = [
        "fnlwgt",
        "education",
        "marital-status",
        "relationship",
        "occupation",
    ],
) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    df.drop(columns=drops, inplace=True)
    return df


def transfrom_dataframe(
    df: pd.DataFrame,
    index: int,
    size: int,
    *,
    seed: Any = None,
) -> tuple[MinMaxScaler, pd.DataFrame, int]:
    df = _select_same(df, index)
    X = df.drop(columns=YCOL)
    y = df[YCOL]

    scaler = _fit_minmax(X)
    X_scaled: pd.DataFrame = scaler.transform(X)  # type: ignore

    kmeans = _fit_kmeans(X_scaled, size, seed)
    X_small = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X_scaled.columns,
    )

    knn = _fit_knn(X_scaled, y, 5)

    pred = knn.predict(X_small)
    y_small = pd.DataFrame(pred, columns=[YCOL])

    df_small = pd.concat((X_small, y_small), axis=1)
    index_small = kmeans.predict(X_scaled.iloc[[index]]).item()

    return scaler, df_small, index_small


sklearn.set_config(transform_output="pandas")

if __name__ == "__main__":
    df = load_dataframe()
    scaler, df_new, index_new = transfrom_dataframe(df, 0, 100, seed=42)
    print(index_new)
