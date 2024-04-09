import math
from functools import partial
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



def cost_fn(df: pd.DataFrame, i: int, j: int) -> list[tuple[float, float]]:
    time = 0.0
    payment = 0.0

    a: pd.Series = df.loc[i]  # type: ignore
    b: pd.Series = df.loc[j]  # type: ignore

    # for age
    time = max(time, b["age"] - a["age"])

    # education
    time = max(time, b["education-num"] - a["education-num"])

    # workclass
    time = max(time, abs(b["workclass"] - a["workclass"]))

    # sigmoid(workclass : hours-per-week)
    eps = 1e-3
    m = a["workclass"] / (a["hours-per-week"] + eps)
    m -= b["workclass"] / (b["hours-per-week"] + eps)
    payment += 1.0 / (1.0 + 1.44 * math.exp(m))  # add bias

    # gain
    temp1 = [(b["capital-gain"] ** 2) - (a["capital-gain"] ** 2), 0]
    temp1[0] = max(temp1[0], time)
    temp1[1] += payment

    temp2 = [
        (b["capital-gain"] + a["capital-gain"]),
        (b["capital-gain"] - a["capital-gain"]),
    ]
    temp2[0] = max(temp2[0], time)
    temp2[1] += payment

    temp3 = [
        (b["capital-gain"] - a["capital-gain"]),
        (b["capital-gain"] + a["capital-gain"]),
    ]
    temp3[0] = max(temp3[0], time)
    temp3[1] += payment

    temp4 = [0, (b["capital-gain"] ** 2) - (a["capital-gain"] ** 2)]
    temp4[0] = max(temp4[0], time)
    temp4[1] += payment

    # loss
    temp1 = [(b["capital-loss"] ** 2) - (a["capital-loss"] ** 2), 0]
    temp1[0] = max(temp1[0], time)
    temp1[1] = payment - temp1[1]

    temp2 = [
        (b["capital-loss"] + a["capital-loss"]),
        (b["capital-loss"] - a["capital-loss"]),
    ]
    temp2[0] = max(temp2[0], time)
    temp2[1] = payment - temp2[1]

    temp3 = [
        (b["capital-loss"] - a["capital-loss"]),
        (b["capital-loss"] + a["capital-loss"]),
    ]
    temp3[0] = max(temp3[0], time)
    temp3[1] = payment - temp3[1]

    temp4 = [
        0,
        (b["capital-loss"] * b["capital-loss"])
        - (a["capital-loss"] * a["capital-loss"]),
    ]
    temp4[0] = max(temp4[0], time)
    temp4[1] = payment - temp4[1]

    return [tuple(temp1), tuple(temp2), tuple(temp3), tuple(temp4)]  # type: ignore


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

    ts = (y == 1).to_numpy().nonzero()[0].tolist()

    graph, dists = recourse(
        X,
        y,
        k,
        s,
        ts,
        partial(cost_fn, df_small),
        limit=limit,
        verbose=False,
    )

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
    main(100, 256, 3, 10, seed=42)
