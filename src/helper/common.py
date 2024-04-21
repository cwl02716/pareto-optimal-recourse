from typing import Any, Unpack

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def select_samples[*T](
    X: pd.DataFrame,
    *args: Unpack[T],
    samples: int,
    seed: int,
    verbose: bool,
) -> tuple[pd.DataFrame, Unpack[T]]:
    samples = min(samples, X.shape[0])
    if verbose:
        print(f"Sampled dataset with size {samples}")
    rng = np.random.default_rng(seed)
    a = np.arange(X.shape[0])
    indices = rng.choice(a, samples, replace=False)
    return select_by_indices(X, *args, indices=indices)


def select_by_mask[*T](
    *args: Unpack[T],
    mask: NDArray,
) -> tuple[Unpack[T]]:
    args = tuple(arg[mask] for arg in args)  # type: ignore
    _reset_index(*args)
    return args


def select_by_indices[*T](
    *args: Unpack[T],
    indices: NDArray,
) -> tuple[Unpack[T]]:
    args = tuple(arg.iloc[indices] for arg in args)  # type: ignore
    _reset_index(*args)
    return args


def _reset_index(*args: Any) -> None:
    return
    for arg in args:
        arg.reset_index(drop=True, inplace=True)
