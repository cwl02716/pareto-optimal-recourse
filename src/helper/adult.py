from pathlib import Path
from typing import Any

import pandas as pd

from helper.common import select_mask

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
    *args: *T,
    index: int,
    verbose: bool,
    immutables: list[Any] = IMMUTABLES,
) -> tuple[pd.DataFrame, *T]:
    X_act = X.drop(columns=immutables)
    X_imm = X[immutables]
    mask = X_imm.eq(X_imm.iloc[index]).all(1).to_numpy()
    res = select_mask(X_act, *args, mask=mask)
    if verbose:
        print(f"Actionable dataset with size {res[0].shape[0]}")
    return res
