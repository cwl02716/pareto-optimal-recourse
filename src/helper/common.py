import numpy as np
import pandas as pd


def get_sample(
    X: pd.DataFrame,
    y: pd.Series,
    size: int,
    *,
    seed: int,
    verbose: bool,
    keep_first: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    if size > X.shape[0]:
        raise ValueError("Sample size is larger than the dataset size")
    rng = np.random.default_rng(seed)
    a = np.arange(1 if keep_first else 0, X.shape[0])
    indices = rng.choice(a, size, replace=False)
    if keep_first:
        np.insert(indices, 0, 0)
    X_sample = X.iloc[indices]
    y_sample = y.iloc[indices]
    X_sample.reset_index(drop=True, inplace=True)
    y_sample.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"Sampled dataset with size {size}")
    return X_sample, y_sample
