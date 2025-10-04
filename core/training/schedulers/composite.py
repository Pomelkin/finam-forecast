from typing import Any
from typing import cast
from typing import override

import torch

from .base import BaseScheduler


class CompositeScheduler(BaseScheduler):
    """
    Composite scheduler that combines multiple schedulers.

    This class is used to manage multiple cosine schedulers under a composite interface.
    It allows combining different schedulers and performing operations like stepping or
    saving/loading states collectively. Each scheduler is identified by a unique key.

    Attributes:
        schedulers (dict[str, Scheduler]): A dictionary of individual schedulers
            where the key is a string identifier and the value is a scheduler instance.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs) -> None:
        for k, v in kwargs.items():
            if not isinstance(v, BaseScheduler):
                raise ValueError(f"Argument '{k}' must be a Scheduler instance.")
        kwargs = cast(dict[str, BaseScheduler], kwargs)
        self.optimizer = optimizer
        self.schedulers = kwargs
        return

    @override
    def state_dict(self) -> dict[str, Any]:
        return {key: value.state_dict() for key, value in self.schedulers.items()}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for key, value in state_dict.items():
            if key in self.schedulers:
                self.schedulers[key].load_state_dict(value)
            else:
                raise ValueError(f"Scheduler {key} not found in state dict.")

    @override
    def step(self, it: int) -> None:
        """
        Updates all schedulers with the given iteration.

        This method iterates through all schedulers stored in the `schedulers` dictionary
        and performs a step operation for each, passing the provided iteration value.

        Args:
            it (int): The current iteration step value to be passed to each scheduler.
        """
        for scheduler in self.schedulers.values():
            scheduler.step(it)

    @override
    def current_value(self) -> dict[str, float]:
        param_state_dict = {}
        for name, scheduler in self.schedulers.items():
            cur_value = scheduler.current_value()
            k, v = next(iter(cur_value.items()))
            param_state_dict[f"{name}_{k}"] = v
        return param_state_dict
