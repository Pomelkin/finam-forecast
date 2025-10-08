from pathlib import Path
from typing import Literal
from typing import TypedDict

import click
import lightning as L
import torch
import torch.distributed as dist
from clearml import InputModel
from clearml import OutputModel
from clearml import Task
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import MixedPrecision

from core.nn import FSDP_SHARD_MODULES
from core.training.callbacks.checkpoint import setup_checkpoint_callback
from core.training.callbacks.early_stopping import setup_early_stopping_callback
from core.training.callbacks.tb_logger import setup_tb_logger
from core.training.configs import DataConfig
from core.training.configs import FSDPStrategyConfig
from core.training.configs import Hyperparams
from core.training.configs import SingleDeviceStrategyConfig
from core.training.configs import TrainingParams
from core.training.data import TimesFMDataModule
from core.training.distributed_utils import is_main_process
from core.training.training_modules import NewsTimesFMTrainingModule
from core.utils import find_version_in_tags
from core.utils import increment_version
from core.utils import setup_logger

logger = setup_logger(add_rank=True)


def _parse_fast_dev_run(fast_dev_run: str) -> int:
    if fast_dev_run == "":
        return 0
    try:
        val = int(fast_dev_run)
        if val < 0:
            raise ValueError()
        return val
    except ValueError:
        raise click.BadParameter(
            f"Invalid value for --fast-dev-run: {fast_dev_run}. It must be a non-negative integer."
        )


def _validate_input(
    remote_execution_queue: str, fast_dev_run: int, profile: bool
) -> None:
    if remote_execution_queue != "" and fast_dev_run > 0:
        raise ValueError(
            "Cannot use `fast-dev-run` with remote execution. Please set `fast-dev-run` to 0 or disable remote execution."
        )

    if profile and remote_execution_queue != "":
        raise ValueError(
            "Cannot use profiling with remote execution. Please disable profiling."
        )

    if profile and fast_dev_run == 0:
        raise ValueError(
            "Cannot use profiling without `fast-dev-run`. Please set `fast-dev-run` to a positive value."
        )
    return


class CallbacksDict(TypedDict):
    checkpoint: ModelCheckpoint
    early_stopping: EarlyStopping | None
    lr_monitor: LearningRateMonitor


def _setup_callbacks(
    task: Task, root_path: Path, training_params: TrainingParams
) -> tuple[list[Callback], CallbacksDict]:
    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_weight_decay=True, log_momentum=True
    )
    checkpoint_callback = setup_checkpoint_callback(
        root_path / "checkpoints" / task.name / task.id,
        training_params.checkpoint,
    )
    if training_params.early_stopping is not None:
        early_stopping_callback = setup_early_stopping_callback(
            training_params.early_stopping
        )
        callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    else:
        early_stopping_callback = None
        callbacks = [checkpoint_callback, lr_monitor]
    return callbacks, {
        "checkpoint": checkpoint_callback,
        "early_stopping": early_stopping_callback,
        "lr_monitor": lr_monitor,
    }


def _setup_loggers(task: Task, root_path: Path) -> list[TensorBoardLogger]:
    loggers = [
        setup_tb_logger(root_path / "runs" / task.name / task.id),
    ]
    return loggers


def _setup_strategy(
    training_params: TrainingParams,
) -> Literal["single_device"] | FSDPStrategy:
    if isinstance(training_params.trainer.devices, list):
        num_devices = len(training_params.trainer.devices)
    else:
        num_devices = training_params.trainer.devices

    if isinstance(training_params.trainer.strategy, FSDPStrategyConfig):
        if num_devices == 1:
            raise ValueError("FSDP strategy requires multiple devices.")

        mixed_precision_config = MixedPrecision(
            param_dtype=getattr(torch, training_params.trainer.strategy.param_dtype),
            reduce_dtype=getattr(torch, training_params.trainer.strategy.reduce_dtype),
            buffer_dtype=getattr(torch, training_params.trainer.strategy.buffer_dtype),
        )
        strategy = FSDPStrategy(
            auto_wrap_policy=FSDP_SHARD_MODULES, mixed_precision=mixed_precision_config
        )
    elif isinstance(training_params.trainer.strategy, SingleDeviceStrategyConfig):
        if num_devices != 1:
            raise ValueError("SingleDevice strategy requires exactly one device.")
        strategy = "single_device"
    else:
        raise ValueError(
            f"Unsupported strategy type: {type(training_params.trainer.strategy)}"
        )
    return strategy


@click.command()
@click.option(
    "--dataset-id", type=click.STRING, required=True, help="ClearML Dataset ID"
)
@click.option(
    "--batch-size", type=click.INT, required=True, help="Batch size for training"
)
@click.option(
    "--num-workers", type=click.INT, required=True, help="Number of DataLoader workers"
)
@click.option(
    "--remote-execution-queue",
    type=click.STRING,
    default="",
    help="Queue for remotely executing task on ClearML. If empty, the training will be run locally.",
)
@click.option(
    "--fast-dev-run",
    type=click.STRING,
    default="0",
    help="Run only a few batches for quick testing. If set to 0 - fast-dev-run will be disabled.",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Enable profiling for the training run.",
    default=False,
)
def train_timesfm(
    dataset_id: str,
    batch_size: int,
    num_workers: int,
    remote_execution_queue: str,
    fast_dev_run: str,
    profile: bool,
) -> None:
    fast_dev_run_ = _parse_fast_dev_run(fast_dev_run)
    _validate_input(remote_execution_queue, fast_dev_run_, profile)
    if fast_dev_run_ == 0:
        fast_dev_run_ = False

    task: Task = Task.init(
        project_name="Finam-FORECAST",
        task_name="NewsTimesFM Fine-tuning",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=True,
        auto_connect_frameworks={
            "pytorch": False,
            "tensorboard": True,
            "matplotlib": True,
            "detect_repository": True,
        },
    )

    ROOT_PATH = Path(__file__).parent.parent.parent

    hyperparams = Hyperparams.connect_as_dict(
        task,
        path=ROOT_PATH / "configs" / "hyperparams.yaml",
    )
    training_params = TrainingParams.connect_as_file(
        task,
        path=ROOT_PATH / "configs" / "training_params.yaml",
    )
    data_cfg = DataConfig(
        dataset_id=dataset_id, batch_size=batch_size, num_workers=num_workers
    )

    if remote_execution_queue != "":
        task.execute_remotely(queue_name="gpu-queue", exit_process=True)

    callbacks, callback_dict = _setup_callbacks(task, ROOT_PATH, training_params)
    loggers = _setup_loggers(task, ROOT_PATH)
    if profile:
        from lightning.pytorch.profilers import SimpleProfiler

        profiler = SimpleProfiler(
            filename=f"{task.name}_{task.id}",
        )
    else:
        profiler = None

    strategy = _setup_strategy(training_params)
    trainer = L.Trainer(
        max_epochs=training_params.trainer.max_epochs,
        accelerator=training_params.trainer.accelerator,
        devices=training_params.trainer.devices,
        strategy=strategy,
        precision=training_params.trainer.precision,
        accumulate_grad_batches=training_params.trainer.accumulate_grad_batches,
        gradient_clip_val=None,
        val_check_interval=training_params.trainer.val_check_interval,
        callbacks=callbacks,
        log_every_n_steps=5,
        logger=loggers,
        fast_dev_run=fast_dev_run_,
        profiler=profiler,
    )

    tokenizer_input_model = InputModel(model_id=training_params.tokenizer_id)
    tokenizer_input_model.connect(task, ignore_remote_overrides=True)
    data_module = TimesFMDataModule(
        data_cfg=data_cfg,
        tokenizer_path=tokenizer_input_model.get_local_copy(),
    )

    clearml_input_model = InputModel(model_id=training_params.model_id)
    clearml_input_model.connect(task, ignore_remote_overrides=True)
    training_module = NewsTimesFMTrainingModule(
        hyperparams=hyperparams,
        path=clearml_input_model.get_local_copy(),
        task=task,
    )

    trainer.fit(training_module, datamodule=data_module)

    if is_main_process():
        tags = clearml_input_model.tags
        if "initialized weights" in tags:
            tags.remove("initialized weights")
            tags.append("fine-tuned weights")

        version = find_version_in_tags(tags)
        if version is None:
            tags.append("v1.0")
        else:
            new_version = increment_version(version)
            tags.remove(version)
            tags.append(new_version)

        output_model = OutputModel(
            task=task,
            name=clearml_input_model.name,
            framework="PyTorch",
            tags=tags,
            config_dict=training_module.model_config,
        )

        ckpt_callback = callback_dict["checkpoint"]
        if ckpt_callback.best_model_path != "":
            logger.info(f"Uploading best model: {ckpt_callback.best_model_path}")
            output_model.update_weights(
                ckpt_callback.best_model_path,
                auto_delete_file=False,
                async_enable=False,
            )

    if dist.is_initialized():
        dist.barrier()
    return


if __name__ == "__main__":
    train_timesfm()
