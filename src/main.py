from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

sklearn.set_config(transform_output="pandas")

# ============================================================
# preprocessing
# ============================================================


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


def select_same_immutable(
    df: pd.DataFrame,
    index: int,
    immutables: list[Any] = [
        "race",
        "sex",
        "native-country",
    ],
) -> pd.DataFrame:
    df_cols = df[immutables]
    i = df_cols.eq(df_cols.iloc[index]).all(1)
    return df[i].drop(columns=immutables)


def minmax(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns="50K")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X["50K"] = df["50K"]
    return X  # type: ignore


def train_knn_model(df: pd.DataFrame) -> KNeighborsClassifier:
    X = df.drop(columns="50K")
    y = df["50K"]
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    return knn


def reduction(df: pd.DataFrame, k: int) -> tuple[KMeans, pd.DataFrame]:
    kmeans = KMeans(k, random_state=0)
    X = df.drop(columns="50K")
    kmeans.fit(X)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    return kmeans, centers


def preprocess(index: int, k: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df = load_dataframe()
    df = select_same_immutable(df, index)
    df = minmax(df)
    knn = train_knn_model(df)
    kmeans, centers = reduction(df, k)
    y_pred = knn.predict(centers)
    centers["50K"] = y_pred
    target = df.drop(columns="50K").iloc[[index]]
    i = kmeans.predict(target).item()
    return df, centers, i


if __name__ == "__main__":
    df, centers, index = preprocess(0, 100)
    print(index)
