from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def select_samples[*T](
    X: pd.DataFrame,
    *args: *T,
    samples: int,
    seed: int,
    verbose: bool,
) -> tuple[pd.DataFrame, *T]:
    samples = min(samples, X.shape[0])
    rng = np.random.default_rng(seed)
    a = np.arange(X.shape[0])
    indices = rng.choice(a, samples, replace=False)
    res = select_indices(X, *args, indices=indices)
    if verbose:
        print(f"Sampled dataset with size {samples}")
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
