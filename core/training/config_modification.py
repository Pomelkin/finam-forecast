from pathlib import Path

import orjson
import torch.distributed as dist

from core.training.distributed_utils import is_main_process
from core.utils import setup_logger

logger = setup_logger(__file__, add_rank=True)


def align_ts_num_input_channels(
    config_path: str | Path, new_num_input_channels: int
) -> None:
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.is_file():
        config_path = config_path / "config.json"

    if config_path.suffix != ".json":
        raise ValueError("Invalid file type. Expected a .json file.")

    if is_main_process():
        config = orjson.loads(config_path.read_bytes())
        if "PatchTST" not in config["architectures"][0]:
            raise ValueError(
                "The provided config does not correspond to a PatchTST model."
            )

        old_num_input_channels = config.get("num_input_channels", None)
        if old_num_input_channels is None:
            raise ValueError(
                "The provided config does not have a 'num_input_channels' field."
            )

        config["num_input_channels"] = new_num_input_channels
        config_path.write_bytes(orjson.dumps(config, option=orjson.OPT_INDENT_2))
        logger.info(
            f"Updated num_input_channels from {old_num_input_channels} to {new_num_input_channels} in {config_path}"
        )
    if dist.is_initialized():
        dist.barrier()
    return
