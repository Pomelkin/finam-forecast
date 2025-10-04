from typing import Any
from typing import override

import numpy as np
import numpy.typing as npt
import torch

from .base import BaseScheduler


class _CosineSchedulerCore(BaseScheduler):
    def __init__(
        self,
        param_group_field: str,
        total_iters: int,
        base_value: float,
        final_value: float,
        warmup_iters_ratio: float | None = None,
        warmup_value: float | None = None,
        freeze_ratio: float | None = None,
    ) -> None:
        if warmup_iters_ratio is not None:
            if not (0 < warmup_iters_ratio < 1):
                raise ValueError(
                    f"Warmup ratio must be in (0, 1), got {warmup_iters_ratio}."
                )
        if (warmup_value is None) != (warmup_iters_ratio is None):
            raise ValueError(
                "Both warmup_ratio and warmup_value must be provided or neither."
            )
        if freeze_ratio is not None:
            if not (0 < freeze_ratio < 1):
                raise ValueError(f"Freeze ratio must be in (0, 1), got {freeze_ratio}.")

        self.param_group_field = param_group_field
        self.total_iters = total_iters
        self.base_value = base_value
        self.final_value = final_value

        self.warmup_iters_ratio = warmup_iters_ratio
        self.warmup_value = warmup_value

        self.freeze_ratio = freeze_ratio

        self.scheduler_values: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.current_value_ = self.base_value
        return

    def _create_scheduler(self) -> None:
        # Create freeze schedule
        if self.freeze_ratio is not None:
            freeze_iters = int(self.total_iters * self.freeze_ratio)
            freeze_schedule = np.zeros(freeze_iters, dtype=np.float64)
        else:
            freeze_iters = 0
            freeze_schedule = np.array([], dtype=np.float64)

        # Create linear warmup schedule
        if self.warmup_iters_ratio is not None and self.warmup_value is not None:
            warmup_iters = int(self.total_iters * self.warmup_iters_ratio)
            warmup_schedule = np.linspace(
                self.warmup_value, self.base_value, warmup_iters, dtype=np.float64
            )
        else:
            if (self.warmup_iters_ratio is None) != (self.warmup_value is None):
                raise ValueError(
                    "Both warmup_value and warmup_iters_ratio must be provided or neither"
                )
            warmup_iters = 0
            warmup_schedule = np.array([], dtype=np.float64)

        cosine_annealing_iters = self.total_iters - warmup_iters - freeze_iters
        if cosine_annealing_iters <= 0:
            raise ValueError("Cosine annealing iters must be > 0.")

        # Create cosine schedule
        iters = np.arange(cosine_annealing_iters)
        schedule = self.final_value + 0.5 * (self.base_value - self.final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )

        # Concatenate all parts of the schedule
        self.scheduler_values = np.concatenate(
            (freeze_schedule, warmup_schedule, schedule)
        )

        if len(self.scheduler_values) != self.total_iters:
            raise ValueError(
                f"Scheduler length ({len(self.scheduler_values)}) does not match total_iters ({self.total_iters})."
            )
        return

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.scheduler_values = np.array([], dtype=np.float64)
        return

    @override
    def step(self, it: int) -> None | float:
        raise NotImplementedError

    def _get_value(self, it: int) -> float:
        if len(self.scheduler_values) == 0:
            self._create_scheduler()

        if it >= self.total_iters:
            value: float = self.final_value
        else:
            value: float = self.scheduler_values[it]
        self.current_value_ = value
        return value

    @override
    def current_value(self) -> dict[str, float]:
        return {self.param_group_field: self.current_value_}


class CosineScheduler(_CosineSchedulerCore):
    """
    Implements a cosine scheduler for adjusting parameter values in torch.optim.Optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_field: str,
        total_iters: int,
        base_value: float,
        final_value: float,
        warmup_iters_ratio: float | None = None,
        warmup_value: float | None = None,
        freeze_ratio: float | None = None,
        multiplier_field: str | None = None,
        skip_if_zero: bool = False,
        apply_if_field: str | None = None,
        ignore_if_field: str | None = None,
    ) -> None:
        """Initialize the scheduler with optimizer and scheduling parameters.

        Args:
            optimizer: PyTorch optimizer to schedule parameters for.
            param_group_field: Name of the parameter group field to modify (e.g., 'lr', 'weight_decay').
            total_iters: Total number of iterations for the scheduling.
            base_value: Initial value for the parameter.
            final_value: Final value for the parameter at the end of scheduling.
            warmup_iters_ratio: Ratio of total iterations to use for warmup phase. Defaults to None.
            warmup_value: Value to use during warmup phase. Defaults to None.
            freeze_ratio: Ratio of total iterations to freeze parameter updates. Defaults to None.
            multiplier_field: Field name for multiplier values in parameter groups. Defaults to None.
            skip_if_zero: Whether to skip scheduling if the parameter value is zero. Defaults to False.
            apply_if_field: Field name that must be present to apply scheduling. Defaults to None.
            ignore_if_field: Field name that when present causes scheduling to be ignored. Defaults to None.
        """
        self.apply_if_field = apply_if_field
        self.ignore_if_field = ignore_if_field
        self.optimizer = optimizer
        self.multiplier_field = multiplier_field
        self.skip_if_zero = skip_if_zero
        super().__init__(
            param_group_field=param_group_field,
            total_iters=total_iters,
            base_value=base_value,
            final_value=final_value,
            warmup_iters_ratio=warmup_iters_ratio,
            warmup_value=warmup_value,
            freeze_ratio=freeze_ratio,
        )
        return

    @override
    def step(self, it: int) -> None:
        value = self._get_value(it)
        for pg in self.optimizer.param_groups:
            if self.param_group_field not in pg:
                raise ValueError(
                    f"Parameter group field '{self.param_group_field}' not found in optimizer parameter groups."
                )

            if (self.apply_if_field is not None) and (self.apply_if_field not in pg):
                continue

            if (self.ignore_if_field is not None) and (self.ignore_if_field in pg):
                continue

            if self.skip_if_zero and pg[self.param_group_field] == 0:
                continue

            if self.multiplier_field is not None:
                if self.multiplier_field not in pg:
                    multiplier = 1.0
                else:
                    multiplier = pg[self.multiplier_field]
                pg[self.param_group_field] = value * multiplier
            else:
                pg[self.param_group_field] = value
        return


class CosineParamScheduler(_CosineSchedulerCore):
    """
    CosineParamScheduler adjusts a parameter value using a cosine annealing scheduler.

    This class provides a mechanism to schedule the value of a parameter over a
    predefined number of iterations. It supports linear warm-up and optional freezing
    periods before the cosine annealing wave begins. The scheduler can be used to
    gradually transition a parameter value from a starting value to a final value.
    """

    @override
    def step(self, it: int) -> float:
        """
        Computes the value corresponding to the given iteration step.
        Args:
            it: The current iteration index used for value computation.

        Returns:
            The computed value for the provided iteration step as a float.
        """
        value = self._get_value(it)
        return value
