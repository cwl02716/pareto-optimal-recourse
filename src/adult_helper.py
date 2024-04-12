from pathlib import Path
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
    df: pd.DataFrame,
    index: int,
    immutables: list[Any] = [],
) -> pd.DataFrame:
    X = df[immutables]
    i = X.eq(X.iloc[index]).all(1)
    df = df[i]
    if immutables:
        df = df.drop(columns=immutables)
    return df
