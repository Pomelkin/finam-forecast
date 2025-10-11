from warnings import warn

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from .config_mixins import ConfigMixin
from core.utils import setup_logger

logger = setup_logger()


class Optimizer(BaseModel):
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999


class Lr(BaseModel):
    use_scheduler: bool = False
    warmup_iters_ratio: float | None = Field(
        default=None, gt=0, lt=1, validate_default=False
    )
    warmup_value: float | None = Field(default=None, gt=0, validate_default=False)
    base_value: float
    final_value: float | None = Field(default=None, gt=0, validate_default=False)

    text_encoder_freeze_iters_ratio: float | None = Field(
        default=None, gt=0, lt=1, validate_default=False
    )
    text_encoder_warmup_iters_ratio: float | None = Field(
        default=None, gt=0, lt=1, validate_default=False
    )

    @model_validator(mode="after")
    def validate_text_encoder_iters(self) -> "Lr":
        freeze_ratio = self.text_encoder_freeze_iters_ratio
        warmup_ratio = self.text_encoder_warmup_iters_ratio
        if (freeze_ratio is None) != (warmup_ratio is None):
            raise ValueError(
                "Both text_encoder_freeze_iters_ratio and text_encoder_warmup_iters_ratio must be provided or neither"
            )
        if (
            (freeze_ratio is not None)
            and (warmup_ratio is not None)
            and not self.use_scheduler
        ):
            logger.warning(
                "use_scheduler is False, text_encoder_freeze_iters_ratio and text_encoder_warmup_iters_ratio will be ignored."
            )
            self.text_encoder_freeze_iters_ratio = None
            self.text_encoder_warmup_iters_ratio = None
        return self

    @model_validator(mode="after")
    def validate_warmup(self) -> "Lr":
        if (self.warmup_value is None) != (
            self.warmup_iters_ratio is None
        ) and self.use_scheduler:
            raise ValueError(
                "Both warmup_value and warmup_iters_ratio must be provided or neither"
            )
        elif (
            (self.warmup_value is not None) or (self.warmup_iters_ratio is not None)
        ) and (not self.use_scheduler):
            logger.warning(
                "use_scheduler is False, warmup_value and warmup_iters_ratio will be ignored."
            )
            self.warmup_value = None
            self.warmup_iters_ratio = None
        return self

    @model_validator(mode="after")
    def validate_final_value(self) -> "Lr":
        if self.use_scheduler and (self.final_value is None):
            raise ValueError("If use_scheduler is True, final_value must be provided.")
        if (not self.use_scheduler) and (self.final_value is not None):
            logger.warning("use_scheduler is False, final_value will be ignored.")
            self.final_value = None
        return self


class WeightDecay(BaseModel):
    use_scheduler: bool = False
    base_value: float
    final_value: float | None = None

    @model_validator(mode="after")
    def validate_final_value(self) -> "WeightDecay":
        if self.use_scheduler and self.final_value is None:
            raise ValueError("If use_scheduler is True, final_value must be provided.")
        if not self.use_scheduler and self.final_value is not None:
            warn("use_scheduler is False, final_value will be ignored.")
        return self


class Hyperparams(ConfigMixin):
    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    optimizer: Optimizer = Optimizer()
    lr: Lr
    weight_decay: WeightDecay

    @model_validator(mode="after")
    def validate_optimizer(self) -> "Hyperparams":
        if self.optimizer is None:
            self.optimizer = Optimizer()
        return self
