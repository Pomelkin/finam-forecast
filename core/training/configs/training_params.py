from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from .config_mixins import ConfigMixin
from core.utils import setup_logger

logger = setup_logger()

PRECISION = Literal[
    64,
    32,
    16,
    "transformer-engine",
    "transformer-engine-float16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
    "64",
    "32",
    "16",
    "bf16",
]


class FSDPStrategyConfig(BaseModel):
    type: Literal["fsdp"]
    param_dtype: Literal["float32", "float16", "bfloat16"]
    reduce_dtype: Literal["float32", "float16", "bfloat16"]
    buffer_dtype: Literal["float32", "float16", "bfloat16"]


class SingleDeviceStrategyConfig(BaseModel):
    type: Literal["single_device"]


class TrainerParameters(BaseModel):
    accelerator: str
    max_epochs: int
    strategy: FSDPStrategyConfig | SingleDeviceStrategyConfig
    val_check_interval: int | float
    devices: list[int] | int
    precision: PRECISION
    accumulate_grad_batches: int = Field(default=1, ge=1)


class EarlyStoppingConfig(BaseModel):
    monitor: str
    mode: str
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.01, ge=0)


class CheckpointConfig(BaseModel):
    save_top_k: int = 4
    monitor: str = "val_loss"
    mode: str = "min"
    filename: str = "{epoch:02d}-{val_loss:.2f}"


class DataConfig(BaseModel):
    dataset_id: str
    batch_size: int
    num_workers: int = Field(ge=1)


class TrainingParams(ConfigMixin):
    trainer: TrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint: CheckpointConfig
    data: DataConfig
