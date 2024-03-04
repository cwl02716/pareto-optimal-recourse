import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path


# ============================================================
# preprocessing
# ============================================================


def load_dataframe(path: str = "dataset/50Ktrain.csv") -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    df.drop(columns=["fnlwgt", "education"], inplace=True)
    return df


def select_same_immutable(df: pd.DataFrame, index: int) -> pd.DataFrame:
    df_col = df[["race", "sex", "native-country"]]
    i = df_col.eq(df_col.iloc[index]).all(1)
    return df_col[i]


def train_knn_model(df: pd.DataFrame) -> KNeighborsClassifier: ...


def predict_labels(model: KNeighborsClassifier, df: pd.DataFrame) -> np.ndarray: ...


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    "StandardScalar"
    ...


def reduction(df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
    """KMeans"""
    ...
