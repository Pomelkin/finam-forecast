import random
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import override

import lightning as L
import plotly.express as px
import torch
import torch.distributed as dist
from clearml import Task
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.regression import MeanAbsolutePercentageError
from torchmetrics.regression import MeanSquaredError

from core.nn import NewsTimesFM_2p5_Model
from core.training.configs import Hyperparams
from core.training.distributed_utils import is_main_process
from core.training.lr_scaling import scale_lrs_for_distributed
from core.training.metrics_formatting import apply_suffix
from core.training.params_groups import create_params_groups
from core.training.schedulers import CompositeScheduler
from core.training.schedulers import CosineScheduler
from core.training.steps_estimation import estimate_total_steps
from core.utils import setup_logger

logger = setup_logger()


def metrics_factory() -> MetricCollection:
    metrics = MetricCollection(
        [
            MeanAbsolutePercentageError(),
            MeanSquaredError(),
            MeanAbsoluteError(),
        ]
    )
    return metrics


class NewsTimesFMTrainingModule(L.LightningModule):
    def __init__(
        self, hyperparams: Hyperparams | dict, path: str | Path, task: Task
    ) -> None:
        super().__init__()
        if isinstance(hyperparams, dict):
            hyperparams = Hyperparams.model_validate(hyperparams)
        self.save_hyperparameters(
            {
                "hyperparams": hyperparams.model_dump(),
            }
        )
        self.training_metrics = metrics_factory()
        self.validation_metrics = metrics_factory()

        self.hyperparams = hyperparams
        self.path = path
        self.loss = nn.MSELoss(reduction="mean")

        self.model: NewsTimesFM_2p5_Model | None = None
        self.task = task
        self.val_outputs: list[dict[str, torch.Tensor]] = []
        return

    @property
    def model_config(self) -> dict:
        if self.model is None:
            raise ValueError("Model is not configured yet.")
        return self.model.config_dict

    @property
    def process_group(self) -> ProcessGroup | None:
        if not dist.is_initialized():
            return None

        if self.device_mesh is not None:
            dp_mesh = self.device_mesh["data_parallel"]
            if dp_mesh.size() == 1:
                logger.warning("Data parallel mesh size is 1, returning None")
                return None
            dp_pg = dp_mesh.get_group()
        else:
            dp_pg = dist.group.WORLD
        return dp_pg

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model must be configured before saving.")
        self.model.save_pretrained(path)
        return

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.model_config
        return

    @override
    def configure_model(self) -> None:
        if self.model is not None:
            return
        self.model = NewsTimesFM_2p5_Model.from_pretrained(self.path, compile=True)
        return

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.model is None:
            raise ValueError("Model must be configured before configuring optimizers.")

        if dist.is_initialized():
            if self.process_group is None:
                raise ValueError("Process group is None, cannot scale learning rates.")
            scale_lrs_for_distributed(
                self.hyperparams.lr,
                group=self.process_group,
            )

        lr_cfg = self.hyperparams.lr
        wd_cfg = self.hyperparams.weight_decay
        optimizer_cfg = self.hyperparams.optimizer

        if lr_cfg.use_scheduler and lr_cfg.warmup_value is not None:
            lr = lr_cfg.warmup_value
        else:
            lr = lr_cfg.base_value

        weight_decay = wd_cfg.base_value
        params_groups = create_params_groups(self.model, weight_decay, lr)

        if optimizer_cfg is not None:
            adamw_beta1 = optimizer_cfg.adamw_beta1
            adamw_beta2 = optimizer_cfg.adamw_beta2
            betas = (adamw_beta1, adamw_beta2)
        else:
            betas = (0.9, 0.999)

        optimizer = AdamW(params_groups, betas=betas)

        total_steps = estimate_total_steps(self.trainer, self.process_group)

        schedulers: dict[str, CosineScheduler] = {}
        if lr_cfg.use_scheduler:
            lr_scheduler = CosineScheduler(
                optimizer,
                param_group_field="lr",
                total_iters=total_steps,
                base_value=lr_cfg.base_value,
                final_value=lr_cfg.final_value,  # type: ignore[assignment]
                warmup_iters_ratio=lr_cfg.warmup_iters_ratio,  # type: ignore[assignment]
                warmup_value=lr_cfg.warmup_value,  # type: ignore[assignment]
                ignore_if_field="loss_param",
            )
            schedulers["lr"] = lr_scheduler
        if wd_cfg.use_scheduler:
            weight_decay_scheduler = CosineScheduler(
                optimizer=optimizer,
                param_group_field="weight_decay",
                total_iters=total_steps,
                base_value=wd_cfg.base_value,
                final_value=wd_cfg.final_value,  # type: ignore[assignment]
                skip_if_zero=True,
            )
            schedulers["weight_decay"] = weight_decay_scheduler

        if len(schedulers) > 1:
            scheduler = CompositeScheduler(
                optimizer=optimizer,
                **schedulers,
            )
        elif len(schedulers) == 1:
            scheduler = list(schedulers.values())[0]
        else:
            scheduler = None

        if scheduler is not None:
            optimizer_config = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
            return optimizer_config  # type: ignore[return-value]
        else:
            return optimizer

    @override
    def lr_scheduler_step(  # type: ignore[override]
        self,
        scheduler: CosineScheduler | CompositeScheduler,
        metric: Any | None,
    ) -> None:
        scheduler.step(self.global_step)
        return

    @override
    def on_before_optimizer_step(self, optimizer) -> None:
        if self.model is None:
            raise ValueError("Model must be configured before optimizer step.")
        if self.hyperparams.grad_clip_val is None:
            return

        if not isinstance(self.trainer.strategy, FSDPStrategy):
            norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.hyperparams.grad_clip_val
            )
        else:
            module: FSDP = self.trainer.strategy.model  # type: ignore
            norm = module.clip_grad_norm_(self.hyperparams.grad_clip_val)

        self.log(
            "grad_norm",
            norm,
            logger=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )
        return

    @override
    def log_dict(
        self,
        dictionary: Mapping[str, Metric | torch.Tensor | int | float]
        | MetricCollection,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | Callable[..., Any] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Any | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        rank_zero_only: bool = False,
        stage: str | None = None,
    ) -> None:
        if stage is not None:
            dictionary = apply_suffix(
                metrics=dictionary,
                suffix=stage,
                add_dist_rank=False,
            )

        super().log_dict(
            dictionary,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            rank_zero_only,
        )
        return

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.model is None:
            raise ValueError("Model must be configured before forward pass.")

        predictions = self.model.forecast(
            inputs_ts=batch["inputs_ts"],
            mask_ts=batch["mask_ts"],
            inputs_text=batch["inputs_text"],
            mask_text=batch["mask_text"],
            targets=batch["targets"],
        )

        outputs = {}
        total_loss = self.loss(
            predictions["normalized_output"],
            predictions["normalized_target"],
        )
        outputs["loss"] = total_loss
        return total_loss, predictions

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, prediction = self(batch)

        metrics: dict[str, torch.Tensor] = self.training_metrics(
            prediction["output"], prediction["target"]
        )
        metrics.update({"loss": loss.detach()})

        self.log_scheduled_values()
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
            stage="train",
        )
        self.log(
            "train_loss",
            metrics["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=False,
            sync_dist=True,
        )
        return loss

    def log_scheduled_values(self) -> None:
        scheduler: CosineScheduler | CompositeScheduler = self.lr_schedulers()  # type: ignore
        scheduler_state_dict = scheduler.current_value()
        scheduler_state_dict = apply_suffix(scheduler_state_dict, "scheduler")
        self.log_dict(
            scheduler_state_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        return

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, prediction = self(batch)

        metrics: dict[str, torch.Tensor] = self.validation_metrics(
            prediction["output"], prediction["target"]
        )
        metrics.update({"loss": loss.detach()})

        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
            stage="val",
        )
        self.log(
            "val_loss",
            metrics["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
        )

        data_to_save = {
            "predictions": prediction["output"][:, -2:].clone().detach().cpu(),
            "targets": prediction["target"][:, -2:].clone().detach().cpu(),
        }
        self.val_outputs.append(data_to_save)
        return loss

    def on_validation_epoch_end(self) -> None:
        if len(self.val_outputs) > 0 and is_main_process():
            val_output = self.val_outputs[random.randint(0, len(self.val_outputs) - 1)]
            predictions = val_output["predictions"]
            targets = val_output["targets"]

            batch_idx = random.randint(0, predictions.size(0) - 1)

            last_preds = predictions[batch_idx].view(-1).float().numpy()
            last_targets = targets[batch_idx].view(-1).float().numpy()

            fig = px.line()
            fig.add_scatter(y=last_preds, mode="lines+markers", name="Prediction")
            fig.add_scatter(y=last_targets, mode="lines+markers", name="Target")

            self.task.get_logger().report_plotly(
                title="Sample Prediction vs Target",
                series="val",
                iteration=self.current_epoch,
                figure=fig,
            )

        if dist.is_initialized():
            dist.barrier()
        self.val_outputs = []
        return
