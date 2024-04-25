import random
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def new_select_samples(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    *,
    rng: random.Random,
    startwith: Sequence[int] = (),
    verbose: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    idx_set = set(X.index.tolist())
    idx_set.difference_update(startwith)
    idx_list = list(startwith) + rng.sample(tuple(idx_set), n_samples - len(startwith))
    if verbose:
        print(f"Sampled dataset with size {n_samples}")
    return X.loc[idx_list], y.loc[idx_list]


def select_samples[*T](
    X: pd.DataFrame,
    *args: *T,
    n_samples: int,
    seed: int,
    verbose: bool,
) -> tuple[pd.DataFrame, *T]:
    n_samples = min(n_samples, X.shape[0])
    rng = np.random.default_rng(seed)
    a = np.arange(X.shape[0])
    indices = rng.choice(a, n_samples, replace=False)
    res = select_indices(X, *args, indices=indices)
    if verbose:
        print(f"Sampled dataset with size {n_samples}")
    return res


def select_mask[*T](*args: *T, mask: NDArray) -> tuple[*T]:
    res = tuple(arg[mask] for arg in args)  # type: ignore
    _reset_index(*res)
    return res  # type: ignore


def select_indices[*T](*args: *T, indices: NDArray) -> tuple[*T]:
    res = tuple(arg.iloc[indices] for arg in args)  # type: ignore
    _reset_index(*res)
    return res  # type: ignore


def _reset_index(*args: Any) -> None:
    pass  # intentionally left blank
