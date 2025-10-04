from pathlib import Path
from typing import TypeVar

import clearml
import yaml
from caseconverter import pascalcase
from caseconverter import snakecase
from pydantic import BaseModel

from core.utils import convert_to_flat_dict
from core.utils import flattened_dict_to_nested

TConfig = TypeVar("TConfig", bound="ConfigMixin")


def load_config(path: Path | str) -> dict:
    """Load a configuration from file."""
    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise ValueError(f"Config file {path} does not exist or is not a file.")

    match path.suffix:
        case ".yaml" | ".yml":
            config = yaml.safe_load(path.open("r"))
        case _:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    return config


class ConfigMixin(BaseModel):
    @classmethod
    def from_file(
        cls: type[TConfig],
        path: str | Path,
    ) -> TConfig:
        """
        Create an instance of the class from a configuration file.

        Args:
            path (str | Path): Path to the configuration file.

        Returns:
            An instance of the class created from the configuration file.
        """
        config = load_config(path)
        instance = cls.model_validate(config)
        return instance

    @classmethod
    def from_dict(
        cls: type[TConfig],
        state_dict: dict,
    ) -> TConfig:
        """
        Creates an instance of the class from a dictionary representation of its
        state.

        Args:
            state_dict (dict): A dictionary representing the state of the
                class that must be validated and used for initialization.

        Returns:
            An initialized instance of the class based on the
                provided state dictionary.
        """
        instance = cls.model_validate(state_dict)
        return instance

    @classmethod
    def connect_as_file(
        cls: type[TConfig],
        task: clearml.Task,
        path: str | Path,
        alias: str | None = None,
    ) -> TConfig:
        """
        Connects as a configuration file to a ClearML Task and initializes an instance of the
        class based on the configuration.

        Args:
            task: The ClearML Task object to which the configuration file
                will be connected, enabling version control and monitoring of configuration
                parameters.
            path (str | Path): Path to the YAML configuration file.
            alias (str | None, optional): An alias for the configuration file in ClearML.
                If None, the alias defaults to the PascalCase version of the class name.

        Returns:
            An instance of the class created from the connected
        """
        if isinstance(path, Path):
            str_path = str(path)
        else:
            str_path = path

        name = alias if alias is not None else pascalcase(cls.__name__)
        connected_path = task.connect_configuration(str_path, name=pascalcase(name))

        if not isinstance(connected_path, str):
            connected_path_str = str(connected_path)
        else:
            connected_path_str = connected_path

        model = cls.from_file(path=connected_path_str)
        return model

    @classmethod
    def connect_as_dict(
        cls: type[TConfig],
        task: clearml.Task,
        path: str | Path,
        alias: str | None = None,
    ) -> TConfig:
        """
        This class method loads configuration from a file as a dictionary, flattens and sync them with ClearML
        task parameters. Then it creates an instance of the class using the synced configuration.

        Args:
            cls: The class type of the model to be created (must be a TRetuningModel subclass).
            task: The ClearML task to connect the configuration to.
            path: Path to the configuration file to load parameters from.
            alias: Optional alias name for the configuration. If None, uses snake_case of class name.

        Returns:
            An instance of the specified class created from the loaded configuration.
        """
        name = alias if alias is not None else snakecase(cls.__name__)

        config = load_config(path)

        flattened_config = convert_to_flat_dict(config)
        task.connect(flattened_config, name=pascalcase(name))
        config = flattened_dict_to_nested(flattened_config)

        model = cls.from_dict(config)
        return model
