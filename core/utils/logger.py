import os
import sys
from pathlib import Path

import logbook
import torch.distributed as dist
from logbook import Logger
from logbook import StreamHandler

handler = StreamHandler(
    sys.stdout,
    format_string="{record.level_name} {record.time:%m-%d %H:%M:%S} [{record.channel}] {record.message}",
)
handler.push_application()


def setup_logger(name: str, add_rank: bool | None = None) -> logbook.Logger:
    """
    Sets up and initializes a logger for distributed or non-distributed environments.

    Args:
        name: The name to be assigned to the logger in non-distributed or single-process
            environments.

    Returns:
        logbook.Logger: An instance of the Logger configured based on the execution context.
    """

    def is_path(value: str) -> bool:
        try:
            p = Path(value)
        except (TypeError, ValueError):
            return False
        return p.exists()

    if is_path(name):
        name = format_filename(name)

    if add_rank is None:
        add_rank = dist.is_initialized()

    if add_rank:
        if "LOCAL_RANK" in os.environ:
            rank = int(os.environ["LOCAL_RANK"])
            logger = Logger(f"rank:{rank} - {name}")
        else:
            logger = Logger(f"rank:{0} - {name}")
    else:
        logger = Logger(name)
    return logger


def format_filename(path: str) -> str:
    return Path(path).name
