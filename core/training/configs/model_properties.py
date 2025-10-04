from pathlib import Path
from typing import Literal

import orjson
import torch.distributed as dist

from .config_mixins import ConfigMixin
from core.training.distributed_utils import is_main_process


class ModelProperties(ConfigMixin):
    text_encoder_id: str
    ts_encoder_id: str
    tokenizer_id: str
    new_num_input_channels: int | None = None
    ts_seq_len: int = 512
    text_max_length: int = 8192

    output_size: int = 384
    normalize_text_features: bool = True
    normalize_timeseries_features: bool = True
    normalization_eps: float = 1e-6
    normalization: Literal["layernorm", "rmsnorm"] | None = "rmsnorm"

    def write_config(self, path: str | Path) -> None:
        """
        Write the model configuration to a config.json file in the specified directory.

        Args:
            path (str | Path): Directory or file path where ``config.json`` will be saved.
        """
        if is_main_process():
            if isinstance(path, str):
                path = Path(path)
            if not path.is_file():
                path = path / "config.json"

            if not path.parent.exists():
                path.parent.mkdir(parents=True)

            state_dict = self.model_dump(
                exclude={
                    "text_encoder_id",
                    "ts_encoder_id",
                    "tokenizer_id",
                    "new_num_input_channels",
                    "ts_seq_len",
                    "text_max_length",
                }
            )
            path.write_bytes(orjson.dumps(state_dict, option=orjson.OPT_INDENT_2))
        if dist.is_initialized():
            dist.barrier()
        return
