import math

import torch.distributed as dist

from core.training.configs.hyperparams import Lr
from core.utils import setup_logger


def scale_lrs_for_distributed(
    lr_config: Lr,
    inv_scale: bool = False,
    group: dist.ProcessGroup | None = None,
    config_name: str = "",
) -> None:
    logger = setup_logger("LR Dist Scaler")

    world_size = dist.get_world_size(group=group)

    if inv_scale:
        scale = 1 / math.sqrt(world_size)
    else:
        scale = math.sqrt(world_size)

    logger.info(f"Scaling learning rates for world size: {world_size}")
    logger.info(f"Scale factor: {scale:.4f}")

    lr_config.base_value *= scale
    logger.info(f"New {config_name} lr BASE: {lr_config.base_value}")

    if lr_config.final_value is not None:
        lr_config.final_value *= scale
        logger.info(f"New {config_name} lr FINAL: {lr_config.final_value}")

    if lr_config.warmup_value is not None:
        lr_config.warmup_value *= scale
        logger.info(f"New {config_name} lr WARMUP: {lr_config.warmup_value}")
    return
