from contextlib import contextmanager
import logging
from typing import Iterator


@contextmanager
def autolog(logger: logging.Logger, name: str) -> Iterator[None]:
    logger.info("start %s", name)
    try:
        yield
    except Exception:
        logger.exception("rrror in %s", name)
        raise
    else:
        logger.info("finish %s", name)
