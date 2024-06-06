import logging
import random
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional

import igraph as ig
import numpy as np
import pandas as pd
import typer
from helper.algorithm import (
    AdditionCost,
    MultiCosts,
    final_costs,
    make_graph,
    multicost_shortest_paths,
)
from helper.dataset import ADULT_IMMUTABLES, load_adult_with_proba
from helper.preproc import batch_loc, batch_mask, get_indices_by_sample

logging.basicConfig(
    format="{asctime} {levelname:<8} {name:<20} {message}",
    datefmt="%H:%M:%S",
    style="{",
    level=logging.DEBUG,
)
logger = logging.getLogger("main")


@dataclass(repr=False, eq=False, frozen=True, slots=True)
class MultiCostsFn:
    X: pd.DataFrame
    columns: list[str]

    @cache
    def __call__(self, i: int, j: int) -> MultiCosts:
        u = self.X.loc[i]
        v = self.X.loc[j]
        return MultiCosts(
            AdditionCost(abs(v.at[c] - u.at[c]).item()) for c in self.columns
        )


def filter_actionable_rows(
    X: pd.DataFrame,
    y: pd.Series,
    index: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X_imm = X[ADULT_IMMUTABLES]
    mask = X_imm.eq(X_imm.loc[index], 1).all(1)
    return batch_mask(mask, (X, y))


def recourse_adult(
    graph: ig.Graph,
    *,
    limit: int = 8,
    key: str = "cost",
) -> list[MultiCosts]:
    dists = multicost_shortest_paths(graph, 0, limit, key=key, verbose=False)
    costs = final_costs(dists)
    return costs


def iter_subgraph(
    X: pd.DataFrame,
    y: pd.Series,
    trials: int,
    samples: Sequence[int],
    index: int,
    k: Any,
    rng: random.Random,
) -> Iterator[ig.Graph]:
    multi_costs_fn = MultiCostsFn(X, ["age", "education-num"])

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
                cost_fn=multi_costs_fn,
                key="cost",
            )

    logger.debug(multi_costs_fn.__call__.cache_info())


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

    logger.info(
        "trails=%s, samples=%s, threshold=%s, seed=%s",
        trials,
        samples,
        threshold,
        seed,
    )

    X_raw, y_proba = load_adult_with_proba()
    logger.info("dataset with %s rows", X_raw.shape[0])

    X_raw, y_proba = filter_actionable_rows(X_raw, y_proba, index)
    logger.info("get %d actionable rows", X_raw.shape[0])

    y = y_proba.gt(threshold).astype(np.int32)
    logger.info("get %d target rows", np.count_nonzero(y.to_numpy()))

    X = X_raw.drop(columns=ADULT_IMMUTABLES)

    with ProcessPoolExecutor() as exe:
        futs = [
            exe.submit(recourse_adult, g)
            for g in iter_subgraph(X, y, trials, samples, index, k, rng)
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


if __name__ == "__main__":
    typer.run(main)
