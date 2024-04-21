from pathlib import Path
from typing import Any, Unpack

import pandas as pd

from helper.common import select_by_mask

PATH = "dataset/50Ktrain.csv"
DROPS = ["fnlwgt", "education", "marital-status", "relationship"]
IMMUTABLES = ["sex", "race", "occupation", "native-country"]
YCOL = "50K"


def load_dataframe(
    path: str = PATH,
    drops: list[Any] = DROPS,
    *,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    if verbose:
        print("Starting reading Adult dataset...")
    df = pd.read_csv(Path(path))
    X = df.drop(columns=YCOL)
    y = df[YCOL]
    if drops:
        X = X.drop(columns=drops)
    if verbose:
        print("Reading Adult dataset finished!")
    return X, y


def select_actionable[*T](
    X: pd.DataFrame,
    *args: Unpack[T],
    index: int,
    immutables: list[Any] = IMMUTABLES,
) -> tuple[pd.DataFrame, Unpack[T]]:
    X_act = X.drop(columns=immutables)
    X_imm = X[immutables]
    mask = X_imm.eq(X_imm.iloc[index]).all(1).to_numpy()
    res = select_by_mask(X_act, *args, mask=mask)
    return res
