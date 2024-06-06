import logging
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from helper.logging import autolog

logger = logging.getLogger(__name__)


ADULT_PATH = Path("dataset/adult.csv")
ADULT_WITH_PROBA_PATH = Path("dataset/adult_proba.csv.gz")
ADULT_MLP_FILE = Path("dataset/adult_mlp.joblib")

ADULT_DROPS = [
    "workclass",
    "fnlwgt",
    "education",
    "marital-status",
    "occupation",
    "relationship",
]
ADULT_IMMUTABLES = [
    "sex",
    "race",
    "native-country",
]
ADULT_CATEGORIES = [
    "sex",
    "race",
    "native-country",
]
ADULT_CONTINUOUS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
ADULT_TRUE = "50K"
ADULT_PROBA = "50K_proba"


def load_dataset(
    path: str | PathLike[str],
    drops: list[Any],
    ycol: str,
    *,
    index_col: Any = None,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, index_col=index_col)
    X = df.drop(columns=ycol)
    y = df[ycol]
    if drops:
        X.drop(columns=drops, inplace=True)
    return X, y


@autolog(logger, "reading Adult dataset")
def load_adult() -> tuple[pd.DataFrame, pd.Series]:
    return load_dataset(
        ADULT_PATH,
        ADULT_DROPS,
        ADULT_TRUE,
    )


@autolog(logger, "reading Adult dataset with probability")
def load_adult_with_proba() -> tuple[pd.DataFrame, pd.Series]:
    return load_dataset(
        ADULT_WITH_PROBA_PATH,
        [],
        ADULT_PROBA,
        index_col=0,
    )
