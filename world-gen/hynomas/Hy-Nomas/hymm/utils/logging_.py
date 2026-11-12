import sys
import pdb
import os

import logging
from loguru import logger
import random


def main_print(content):
    if int(os.environ["RANK"]) <= 0:
        print(content)


def random_print(content, random_prob=0.01):
    if int(os.environ["RANK"]) <= 0 and random.random() < random_prob:
        print(content)


# ForkedPdb().set_trace()
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def setup_logger(exp_dir):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if rank <= 0:
        # Remove any previously added file sinks to avoid duplicate handlers
        # on resume (loguru's logger.add() is cumulative).
        logger.remove()
        # Re-add stderr sink (loguru removes it when logger.remove() is called)
        logger.add(sys.stderr, level="DEBUG")

        train_logger = logger.bind(name="train")
        val_logger = logger.bind(name="val")
    else:
        # Non-zero ranks still need dataloader warnings/errors; suppress info/debug
        # to avoid flooding logs during large distributed runs.
        logger.remove()
        logger.add(
            sys.stderr,
            level="WARNING",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                f"rank={rank} local_rank={local_rank} | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
        train_logger = logger.bind(name="train")
        val_logger = logger.bind(name="val")

    train_logger.info(f"Experiment directory created at: {exp_dir}")

    return train_logger, val_logger
