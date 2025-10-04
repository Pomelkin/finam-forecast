import copy
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import override

import lightning as L
import torch
import torch.distributed as dist
from clearml import OutputModel
from clearml import Task
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.optim import AdamW
from torchmetrics import Metric
from torchmetrics import MetricCollection

from core.nn import FinClipModel
from core.training.configs import Hyperparams
from core.training.loss import ClipLoss
from core.training.loss import SigLipLoss
from core.training.lr_scaling import scale_lrs_for_distributed
from core.training.metrics_formatting import apply_suffix
from core.training.params_ops import create_params_groups
from core.training.schedulers import CompositeScheduler
from core.training.schedulers import CosineScheduler
from core.training.steps_estimation import estimate_total_steps
from core.utils import setup_logger

logger = setup_logger(__file__)


class FinClipTrainingModule(L.LightningModule):
    def __init__(
        self, hyperparams: Hyperparams | dict, model_path: str | Path, task: Task
    ) -> None:
        super().__init__()
        if isinstance(hyperparams, dict):
            hyperparams = Hyperparams.model_validate(hyperparams)
        self.save_hyperparameters(
            {
                "hyperparams": hyperparams.model_dump(),
            }
        )

        match hyperparams.loss_type:
            case "clip":
                self.loss = ClipLoss()
            case "siglip":
                self.loss = SigLipLoss()
            case _:
                raise ValueError(f"Unknown loss type: {hyperparams.loss_type}")

        self.hyperparams = hyperparams
        self.model_path = model_path
        self.task = task

        self.model: FinClipModel | None = None
        self.clearml_output_models: list[OutputModel] = []
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
        self.model = FinClipModel.from_pretrained(self.model_path, strict=False)
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
                inv_scale=True,
                group=self.process_group,
                config_name="model",
            )
            scale_lrs_for_distributed(
                self.hyperparams.loss_lr,
                inv_scale=False,
                group=self.process_group,
                config_name="loss",
            )

        lr_cfg = self.hyperparams.lr
        loss_lr_cfg = self.hyperparams.loss_lr
        wd_cfg = self.hyperparams.weight_decay
        optimizer_cfg = self.hyperparams.optimizer

        if lr_cfg.use_scheduler and lr_cfg.warmup_value is not None:
            lr = lr_cfg.warmup_value
        else:
            lr = lr_cfg.base_value

        if loss_lr_cfg.use_scheduler and loss_lr_cfg.warmup_value is not None:
            loss_lr = loss_lr_cfg.warmup_value
        else:
            loss_lr = loss_lr_cfg.base_value

        weight_decay = wd_cfg.base_value
        params_groups = create_params_groups(
            self.model, weight_decay, lr, self.loss, loss_lr
        )

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
        if loss_lr_cfg.use_scheduler:
            loss_lr_scheduler = CosineScheduler(
                optimizer,
                param_group_field="lr",
                total_iters=total_steps,
                base_value=loss_lr_cfg.base_value,
                final_value=loss_lr_cfg.final_value,  # type: ignore[assignment]
                warmup_iters_ratio=loss_lr_cfg.warmup_iters_ratio,  # type: ignore[assignment]
                warmup_value=loss_lr_cfg.warmup_value,  # type: ignore[assignment]
                apply_if_field="loss_param",
            )
            schedulers["loss_lr"] = loss_lr_scheduler
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

        if dist.is_initialized():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.grad.mul_(dist.get_world_size())
                    logger.info(
                        f"Parameter {name} gradient scaled. Params size: {param.shape}, dtype: {param.grad.dtype}, device: {param.grad.device}"
                    )
                else:
                    logger.warning(f"Parameter {name} has no gradient.")

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

            if stage == "test":
                self.log_to_output_models(dictionary)

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

    def add_output_model(self, output_model: OutputModel) -> None:
        self.clearml_output_models.append(output_model)
        return

    def log_to_output_models(
        self,
        metrics: Mapping[str, Metric | torch.Tensor | int | float | DTensor]
        | MetricCollection,
    ) -> None:
        metrics = copy.deepcopy(metrics)
        if isinstance(metrics, MetricCollection):
            processed_metrics = metrics.compute()
        else:
            processed_metrics: Mapping[str, torch.Tensor] = {}
            for k, v in metrics.items():
                if isinstance(v, Metric):
                    proc_v = v.compute().to(self.device)
                elif isinstance(v, DTensor):
                    proc_v = v.to_local().to(self.device)
                elif isinstance(v, float | int):
                    proc_v = torch.tensor(v, dtype=torch.float32, device=self.device)
                elif isinstance(v, torch.Tensor):
                    proc_v = v.to(self.device)
                else:
                    raise TypeError(
                        f"Unsupported metric type: {type(v)} for key: {k}. "
                        "Expected torch.Tensor, float, or int."
                    )
                processed_metrics[k] = proc_v

        if len(self.clearml_output_models) != 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                for output_model in self.clearml_output_models:
                    if not isinstance(output_model, OutputModel):
                        continue
                    for key, value in processed_metrics.items():
                        value = value.item()
                        output_model.report_scalar(
                            title="Metrics",
                            series=key,
                            value=value,
                            iteration=self.model_logger_counter,
                        )
                self.model_logger_counter += 1

        if dist.is_initialized():
            dist.barrier()
        return

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ValueError("Model must be configured before forward pass.")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        past_values = batch["past_values"]
        past_observed_mask = batch["past_observed_mask"]

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
        )
        loss, B = self.loss(
            text_features=output["text_features"],
            timeseries_features=output["timeseries_features"],
            process_group=self.process_group,
        )
        return {"loss": loss, "B": B}

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch)
        loss = outputs["loss"]
        B = outputs["B"]
        d_loss = loss.detach()

        self.log_scheduled_values()
        self.log_dict(
            {"loss": d_loss},
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
            stage="train",
            batch_size=B,
        )
        self.log(
            "train_loss",
            d_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            batch_size=B,
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

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch)

        loss = outputs["loss"]
        B = outputs["B"]
        d_loss = loss.detach()

        self.log_dict(
            {"loss": d_loss},
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
            stage="val",
            batch_size=B,
        )
        self.log(
            "val_loss",
            d_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            batch_size=B,
        )
        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch)

        loss = outputs["loss"]
        B = outputs["B"]
        d_loss = loss.detach()

        self.log_dict(
            {"loss": d_loss},
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
            stage="test",
            batch_size=B,
        )
        self.log(
            "test_loss",
            d_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            batch_size=B,
        )
        return loss
