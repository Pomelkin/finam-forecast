from pathlib import Path
from shutil import rmtree

from lightning.pytorch.loggers import TensorBoardLogger

from core.training.distributed_utils import is_main_process
from core.utils import setup_logger


def setup_tb_logger(
    runs_dir: Path,
) -> TensorBoardLogger:
    logger = setup_logger("TensorBoard Logger Setup", add_rank=True)

    if runs_dir.exists():
        if is_main_process():
            logger.warn(f"TensorBoard log directory {runs_dir} already exists.")
            rmtree(runs_dir)
            logger.warn(f"Removed existing TensorBoard log directory {runs_dir}.")
    else:
        logger.info(f"Creating TensorBoard log directory {runs_dir}.")
        runs_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=runs_dir,
        name="tb_logs",
        default_hp_metric=False,
    )
    return tb_logger
