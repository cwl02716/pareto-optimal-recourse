import logging
import random
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import cache, partial
from pathlib import Path
from typing import Annotated, Any, Optional, cast

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
from helper.dataset import ADULT_CONTINUOUS, ADULT_IMMUTABLES, load_adult_with_proba
from helper.preproc import batch_loc, batch_mask, get_indices_by_sample
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    format="{asctime} {levelname:<8} {name:<20} {message}",
    datefmt="%H:%M:%S",
    style="{",
    level=logging.DEBUG,
)
logger = logging.getLogger("main")

COST_CLOUMNS = ["age", "education-num"]


@dataclass(repr=False, eq=False, frozen=True, slots=True)
class MultiCostsFn:
    X: pd.DataFrame

    @cache
    def __call__(self, i: int, j: int) -> MultiCosts:
        u = self.X.loc[i]
        v = self.X.loc[j]
        return MultiCosts(
            AdditionCost(abs(v.at[c] - u.at[c]).item()) for c in COST_CLOUMNS
        )


def recourse_adult(
    graph: ig.Graph,
    *,
    limit: int,
    key: str,
) -> list[tuple[float, ...]]:
    dists = multicost_shortest_paths(graph, 0, limit, key=key, verbose=False)
    costs = [(*map(float, c),) for c in final_costs(dists)]
    return costs


def iter_subgraph(
    X: pd.DataFrame,
    y: pd.Series,
    trials: int,
    samples: Sequence[int],
    index: int,
    k: Any,
    *,
    key: str,
    rng: random.Random,
    multi_costs_fn: MultiCostsFn,
) -> Iterator[tuple[int, int, ig.Graph]]:
    for t in range(trials):
        Xi = X
        yi = y
        for n in samples:
            indices = get_indices_by_sample(
                n, Xi.index.to_list(), startwith=index, rng=rng
            )
            Xi, yi = batch_loc(indices, (Xi, yi))
            targets = yi.to_numpy().nonzero()[0].tolist()
            n_targets = len(targets)

            logger.info("trial #%d with %d samples and %d targets", t, n, n_targets)

            g = make_graph(
                Xi,
                targets,
                k,
                key=key,
                cost_fn=multi_costs_fn,
                maker_fn=partial(kneighbors_graph, n_jobs=-1),
            )

            ecount = g.ecount()

            if ecount == 0:
                continue

            if ecount <= 2 * g.vcount():
                logger.warning("graph with %d edges is too small", ecount)

            yield t, n, g

    logger.debug(multi_costs_fn.__call__.cache_info())


def main(
    trials: Annotated[
        int,
        typer.Argument(help="number of trials"),
    ],
    samples: Annotated[
        list[int],
        typer.Argument(help="list of samples in each trial"),
    ],
    index: Annotated[
        int,
        typer.Option("--index", "-i", help="loc of target row"),
    ] = 0,
    k: int = 5,
    threshold: float = 0.75,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> None:
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    rng = random.Random(seed)
    samples = sorted(samples, reverse=True)

    logger.info(
        "trials=%s, samples=%s, threshold=%s, seed=%s",
        trials,
        samples,
        threshold,
        seed,
    )

    X_raw, y_proba = load_adult_with_proba()
    logger.info("dataset with %s rows", X_raw.shape[0])

    index_proba = y_proba.at[index].item()
    logger.info("row #%d has probability %.2f", index, index_proba)

    if index_proba >= threshold:
        logger.warning("row #%d is already target", index)
        return

    # get actionable rows mask
    X_imm = X_raw[ADULT_IMMUTABLES]
    mask = X_imm.eq(X_imm.loc[index], 1).all(1)

    scaler = MinMaxScaler()
    scaler.set_output(transform="pandas")
    X_scaled = cast(pd.DataFrame, scaler.fit_transform(X_raw[ADULT_CONTINUOUS]))

    X_scaled, y_proba = batch_mask(mask, (X_scaled, y_proba))

    logger.info("get %d actionable rows", y_proba.shape[0])

    y_label = y_proba.gt(threshold)
    logger.info("get %d target rows", np.count_nonzero(y_label.to_numpy()))

    multi_costs_fn = MultiCostsFn(X_raw)

    with ProcessPoolExecutor() as exe:
        futs = {
            (t, n): exe.submit(
                recourse_adult,
                g,
                limit=8,
                key="cost",
            )
            for t, n, g in iter_subgraph(
                X_scaled,
                y_label,
                trials,
                samples,
                index,
                k,
                key="cost",
                rng=rng,
                multi_costs_fn=multi_costs_fn,
            )
        }

    data = [
        (t, n, i, *cost)
        for (t, n), fut in futs.items()
        for i, cost in enumerate(fut.result())
    ]

    df = pd.DataFrame(
        data,
        columns=("#trial", "samples", "#result", *COST_CLOUMNS),
    )

    parent_dir = Path("dataset/adult_epsilon")

    if not parent_dir.exists():
        parent_dir.mkdir(parents=True)
        logger.warning("create directory %s since it does not exist", parent_dir)

    path = parent_dir / f"{datetime.now():%Y%m%dT%H%M%S}.csv"

    df.to_csv(path, index=False)

    logger.info("write %d results to %s", len(data), path)


if __name__ == "__main__":
    typer.run(main)
