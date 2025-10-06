from pathlib import Path
from shutil import rmtree

from lightning.pytorch.callbacks import ModelCheckpoint

from core.training.configs.training_params import CheckpointConfig
from core.training.distributed_utils import is_main_process
from core.utils import setup_logger


def setup_checkpoint_callback(
    dirpath: Path,
    ckpt_cfg: CheckpointConfig,
    save_weights_only: bool = True,
) -> ModelCheckpoint:
    logger = setup_logger("Checkpoint Callback Setup", add_rank=True)

    if dirpath.exists():
        if is_main_process():
            logger.warning(f"Checkpoint directory {dirpath} already exists.")
            rmtree(dirpath)
            logger.warning(f"Removed existing checkpoint directory {dirpath}.")
    else:
        logger.info(f"Creating checkpoint directory {dirpath}.")
        dirpath.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename=ckpt_cfg.filename,
        save_top_k=ckpt_cfg.save_top_k,
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        verbose=True,
        save_weights_only=save_weights_only,
    )
    return checkpoint_callback
