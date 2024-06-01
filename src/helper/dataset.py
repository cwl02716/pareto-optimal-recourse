import logging
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from helper.logging import autolog

logger = logging.getLogger(__name__)


ADULT_PATH = Path("dataset/50Ktrain.csv")

ADULT_DROPS = ["fnlwgt", "education", "marital-status", "relationship"]
ADULT_IMMUTABLES = ["sex", "race", "occupation", "native-country"]
ADULT_YCOL = "50K"


@autolog(logger, "reading Adult dataset")
def load_adult(
    path: str | PathLike[str] = ADULT_PATH,
    drops: list[Any] = ADULT_DROPS,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=ADULT_YCOL)
    y = df[ADULT_YCOL]
    if drops:
        X = X.drop(columns=drops)
    return X, y
