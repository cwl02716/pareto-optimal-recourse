import logging
import random
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Optional

import igraph as ig
import numpy as np
import pandas as pd
import sklearn
import typer
from helper.algorithm import (
    AdditionCost,
    MultiCost,
    final_costs,
    make_graph,
    multicost_shortest_paths,
)
from helper.dataset import ADULT_IMMUTABLES, load_adult_with_proba
from helper.preproc import batch_loc, batch_mask, get_indices_by_sample

logging.basicConfig(
    format="{asctime} {levelname:<8} {name:<24} {message}",
    datefmt="%H:%M:%S",
    style="{",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
sklearn.set_config(transform_output="pandas")


def multi_costs_fn(X: pd.DataFrame, cols: list[str], i: int, j: int) -> MultiCost:
    a = X.iloc[i]
    b = X.iloc[j]
    return MultiCost(
        tuple(AdditionCost(abs(b.at[c].item() - a.at[c].item())) for c in cols)
    )


def main(
    trials: int,
    samples: list[int],
    index: int = 0,
    k: int = 4,
    threshold: float = 0.75,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> None:
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    rng = random.Random(seed)
    samples = sorted(samples, reverse=True)

    logger.info("trails=%s, samples=%s, seed=%s", trials, samples, seed)

    X_raw, y_proba = load_adult_with_proba()
    X_raw, y_proba = filter_actionable_rows(X_raw, y_proba, index)

    logger.info("get %s actionable rows", X_raw.shape[0])

    X = X_raw.drop(columns=ADULT_IMMUTABLES)
    y = y_proba.gt(threshold).astype(np.int32)

    logger.info(
        "set target threshold to %.3g, get %d targets",
        threshold,
        y.value_counts()[1],
    )

    with ProcessPoolExecutor() as exe:
        futs = [
            exe.submit(recourse_adult, g)
            for g in iter_subsample(X, y, trials, samples, index, k, rng)
        ]

    res = [fut.result() for fut in futs]

    logger.info(res)

    # data = []
    # for (n, t), future in zip(product(range(trials), samples), as_completed(futs)):
    #     if verbose:
    #         print(f"complete trial: {t}, n_samples: {n}")
    #     for i, costs in enumerate(future.result()):
    #         data.append((n, t, i, *map(float, costs)))

    # df = pd.DataFrame(
    #     data,
    #     columns=("trial", "n_samples", "result", "cost_1", "cost_2"),
    # )

    # df.to_csv("dataset/mnist_epsilon.csv", index=False)


def filter_actionable_rows(
    X: pd.DataFrame,
    y: pd.Series,
    index: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X_imm = X[ADULT_IMMUTABLES]
    mask = X_imm.eq(X_imm.loc[index], 1).all(1)
    return batch_mask(mask, (X, y))


def iter_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    trials: int,
    samples: Sequence[int],
    index: int,
    k: Any,
    rng: random.Random,
) -> Iterator[ig.Graph]:
    for t in range(trials):
        Xi = X
        yi = y
        for n in samples:
            indices = get_indices_by_sample(n, Xi.index, startwith=index, rng=rng)
            Xi, yi = batch_loc(indices, (Xi, yi))
            targets = yi.to_numpy().nonzero()[0].tolist()

            logger.info("trial #%d with %d samples and %d targets", t, n, len(targets))

            yield make_graph(
                Xi,
                targets,
                k,
                cost_fn=partial(multi_costs_fn, Xi, ["age", "education-num"]),
                key="cost",
            )


def recourse_adult(
    graph: ig.Graph,
    limit: int = 8,
    *,
    key: str = "cost",
) -> list[MultiCost]:
    dists = multicost_shortest_paths(graph, 0, limit, key=key, verbose=False)
    costs = final_costs(dists)
    return costs


if __name__ == "__main__":
    typer.run(main)
