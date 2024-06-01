from contextlib import contextmanager
import logging
from typing import Iterator


@contextmanager
def autolog(logger: logging.Logger, name: str) -> Iterator[None]:
    logger.info("Start %s", name)
    try:
        yield
    except Exception:
        logger.exception("Error in %s", name)
        raise
    else:
        logger.info("Finish %s", name)
