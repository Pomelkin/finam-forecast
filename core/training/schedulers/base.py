from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseScheduler(ABC):
    def state_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "scheduler_values"]
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def __getstate__(self) -> dict[str, Any]:
        return self.state_dict()

    def __setstate__(self, state) -> None:
        self.load_state_dict(state)
        return

    @abstractmethod
    def step(self, it: int) -> None | float:
        """Update the scheduler state. This is a no-op for most schedulers.
        Args:
            it (int): The current iteration step.
        Returns:
            The updated value for the parameter, if applicable.
        """
        pass

    @abstractmethod
    def current_value(self) -> dict[str, float]:
        pass
