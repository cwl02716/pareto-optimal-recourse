import pandas as pd


def get_sample(
    X: pd.DataFrame, y: pd.Series, size: int, *, seed: int, verbose: bool
) -> tuple[pd.DataFrame, pd.Series]:
    X_sample = X.sample(size, random_state=seed)
    y_sample = y[X_sample.index]
    X_sample.reset_index(drop=True, inplace=True)
    y_sample.reset_index(drop=True, inplace=True)
    if verbose:
        print(f"Sampled dataset with size {size}")
    return X_sample, y_sample