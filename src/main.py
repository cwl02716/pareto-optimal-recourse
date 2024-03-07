from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# preprocessing
# ============================================================


def load_dataframe(
    path: str = "dataset/50Ktrain.csv",
    drops: list[Any] = ["fnlwgt", "education"],
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
    drops: list[Any] = [
        "race",
        "sex",
        "native-country",
        "occupation",
        "relationship",
    ],
) -> pd.DataFrame:
    df_cols = df[immutables]
    i = df_cols.eq(df_cols.iloc[index]).all(1)
    return df[i].drop(columns=drops)


def train_knn_model(df: pd.DataFrame) -> KNeighborsClassifier:
    X = df.drop(columns="50K")
    y = df["50K"]
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    return knn


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    scaler.fit_transform(df[:-1], copy=False)
    return df


def reduction(df: pd.DataFrame, k: int) -> pd.DataFrame:
    kmeans = KMeans(k, random_state=0)
    X = df.drop(columns="50K")
    kmeans.fit(X)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    centers["50K"] = df["50K"]
    return centers


def predict(df: pd.DataFrame, model: KNeighborsClassifier) -> pd.DataFrame:
    X = df.drop(columns="50K")
    return model.predict(X) # type: ignore


if __name__ == "__main__":

    def main() -> None:
        df = load_dataframe()
        df = select_same_immutable(df, 0)
        knn = train_knn_model(df)
        centers = reduction(df, 100)
        res = predict(centers, knn)
        print(res)

    sklearn.config_context(transform_output="pandas")
    main()
