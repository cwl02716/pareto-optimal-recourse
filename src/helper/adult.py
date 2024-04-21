from pathlib import Path
import random
from typing import Any

import pandas as pd

__all__ = ["load_dataframe", "select_rows_by_immutables"]


def load_dataframe(
    path: str,
    drops: list[Any] = [],
) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    if drops:
        df = df.drop(columns=drops)
    return df


def select_rows_by_immutables(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    immutables: list[Any] = [],
) -> tuple[pd.DataFrame, pd.Series]:
    rng = random.Random(seed)
    index = rng.sample((y == 1).to_numpy().nonzero()[0].tolist(), 1)[0]

    X_act = X.drop(columns=immutables)
    X_imm = X[immutables]
    mask = X_imm.eq(X_imm.iloc[index]).all(1)
    mask[index] = False
    X_sample = pd.concat((X_act.iloc[[index]], X_act[mask]), axis=0, ignore_index=True)
    y_sample = pd.concat((pd.Series(y.iat[index]), y[mask]), axis=0, ignore_index=True)
    return X_sample, y_sample
