from contextlib import contextmanager
import logging
from typing import Iterator


@contextmanager
def autolog(
    logger: logging.Logger,
    name: str,
    level: int = logging.DEBUG,
) -> Iterator[None]:
    logger.log(level, "start %s", name)
    try:
        yield
    except Exception:
        logger.exception("rrror in %s", name)
        raise
    else:
        logger.log(level, "finish %s", name)
