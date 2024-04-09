import math

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


train_samples = 5000

X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False
)  # keep numpy.ndarray
scaler = StandardScaler()



def costs(df: np.ndarray, i: int, j: int) -> list[tuple[float, float]]:
    d1 = 0.0  # l1 distance
    d2 = 0.0  # l2 distance

    a: pd.Series = df.iloc[i]  # type: ignore
    b: pd.Series = df.iloc[j]  # type: ignore
    m: pd.Series = df.iloc[[i, j]]  # type: ignore
    diff = a - b
    d1 = np.linalg.norm(ord=1, x=diff)
    d2 = np.linalg.norm(ord=2, x=diff)
    # 取最大值of m
    m0 = m.max(axis=1)  # type: ignore

    d2 = math.sqrt(d2.sum())
    d1 /= m0
    d2 /= m0
    return [(d1, d2)]  # type: ignore
