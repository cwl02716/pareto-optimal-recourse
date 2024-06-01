"""new submodule for preprocessing data, intended to replace the common submodule"""

import random
from typing import Iterable

import pandas as pd
from numpy.typing import ArrayLike, NDArray

DEFAULT_RNG = random.Random()


def batch_loc[*T](indices: ArrayLike, args: tuple[*T]) -> tuple[*T]:
    return tuple(arg.loc[indices] for arg in args)  # type: ignore


def batch_iloc[*T](indices: ArrayLike, args: tuple[*T]) -> tuple[*T]:
    return tuple(arg.iloc[indices] for arg in args)  # type: ignore


def batch_mask[*T](mask: pd.Series, args: tuple[*T]) -> tuple[*T]:
    return tuple(arg[mask] for arg in args)  # type: ignore


def get_indices_by_sample(
    n_samples: int,
    population: Iterable[int],
    *,
    startwith: int,
    rng: random.Random = DEFAULT_RNG,
) -> list[int]:
    idx_set = set(population)
    idx_set.remove(startwith)
    idx_list = [startwith]
    idx_list += rng.sample(tuple(idx_set), n_samples - 1)
    return idx_list


def get_targets(y: NDArray, threshold: float) -> list[int]:
    return (y >= threshold).nonzero()[0].tolist()


def get_mask_by_eq(X: pd.DataFrame, index: int) -> pd.Series:
    return X.eq(X.loc[index]).all(1)
