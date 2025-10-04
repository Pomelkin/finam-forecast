import os

import torch.distributed as dist


def _get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return _get_rank() == 0
