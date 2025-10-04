# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TimesFM models."""

from pathlib import Path

import orjson
import torch
from safetensors.torch import load_file
from safetensors.torch import save_file
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from .configs import TimesFM_2p5_200M_Config
from core.nn.text_encoder.modernbert import ModernBertConfig
from core.nn.text_encoder.modernbert import ModernBertModel
from core.nn.timesfm.layers import compute_causal_statistics
from core.nn.timesfm.layers import ResidualBlock
from core.nn.timesfm.layers import revin
from core.nn.timesfm.layers import Transformer
from core.utils import setup_logger


logger = setup_logger(__file__)


def log_incompatible_keys(incompatible_keys: _IncompatibleKeys) -> None:
    if incompatible_keys.missing_keys:
        logger.warning(f"Missing keys: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        logger.warning(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
    return


class TimesFM_2p5_Model(nn.Module):
    """TimesFM 2.5 with 200M parameters."""

    def __init__(self, text_encoder_config: ModernBertConfig) -> None:
        super().__init__()
        config = TimesFM_2p5_200M_Config()

        self.text_encoder_config = text_encoder_config
        self.text_encoder = ModernBertModel(text_encoder_config)
        self.text_projection = nn.Linear(
            text_encoder_config.hidden_size,
            config.stacked_transformers.transformer.model_dims,
        )

        # Names constants.
        self.input_patch_len = config.input_patch_len  # 32
        self.output_patch_len = config.output_patch_len  # 128
        self.m = self.output_patch_len // self.input_patch_len  # 4
        self.num_layers = config.stacked_transformers.num_layers  # 20
        self.num_heads = config.stacked_transformers.transformer.num_heads  # 16
        self.model_dims = config.stacked_transformers.transformer.model_dims  # 1280
        self.num_heads = self.model_dims // self.num_heads  # 80

        # Layers.
        self.tokenizer = ResidualBlock(config.tokenizer)
        self.stacked_xf: nn.ModuleList = nn.ModuleList(
            [
                Transformer(config.stacked_transformers.transformer)
                for _ in range(self.num_layers)
            ]
        )
        self.output_projection_point = ResidualBlock(config.output_projection_point)
        return

    @property
    def config_dict(self) -> dict:
        config = {"text_encoder": self.text_encoder_config.to_dict()}
        return config

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
        input_embeddings = self.tokenizer(tokenizer_inputs)

        output_embeddings = input_embeddings
        for layer in self.stacked_xf:
            output_embeddings = layer(output_embeddings, masks[..., -1])
        output_ts = self.output_projection_point(output_embeddings)
        return output_ts

    def forecast(self, inputs: torch.Tensor, mask: torch.Tensor):
        batch_size, context_len = inputs.shape[0], inputs.shape[1]

        if (pad_len := -(context_len) % self.input_patch_len) != 0:
            inputs = torch.cat(
                [
                    torch.zeros(
                        batch_size,
                        pad_len,
                        device=inputs.device,
                    ),
                    inputs,
                ],
                dim=1,
            )
            mask = torch.cat(
                [
                    torch.ones(
                        batch_size,
                        pad_len,
                        device=mask.device,
                        dtype=torch.bool,
                    ),
                    mask,
                ],
                dim=1,
            )
            context_len += pad_len

        causal_means, causal_scale = compute_causal_statistics(inputs, mask)
        normalized_inputs = revin(inputs, causal_means, causal_scale, reverse=False)
        normalized_output_ts = self(normalized_inputs, mask)
        output_ts = revin(
            normalized_output_ts, causal_means, causal_scale, reverse=True
        )
        output_ts = torch.reshape(output_ts, (batch_size, -1, self.output_patch_len))
        return output_ts

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        compile: bool = False,
    ) -> "TimesFM_2p5_Model":
        if device is None:
            device = torch.get_default_device()
        if dtype is None:
            dtype = torch.get_default_dtype()

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory")

        text_encoder_config_dict = orjson.loads(
            (path / "text_encoder_config.json").read_bytes()
        )
        text_encoder_config = ModernBertConfig.from_dict(text_encoder_config_dict)
        cls = cls(text_encoder_config=text_encoder_config).to(
            dtype=dtype, device=device
        )

        state_dict = load_file(path / "model.safetensors", device=str(device))
        incompatible_keys = cls.load_state_dict(state_dict)
        log_incompatible_keys(incompatible_keys)
        if compile:
            cls = torch.compile(cls, mode="reduce-overhead")
        return cls  # type: ignore

    @classmethod
    def from_config(
        cls,
        config: dict,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        compile: bool = False,
    ) -> "TimesFM_2p5_Model":
        text_encoder_config = ModernBertConfig.from_dict(config["text_encoder"])
        cls = cls(text_encoder_config=text_encoder_config).to(
            dtype=dtype, device=device
        )
        if compile:
            cls = torch.compile(cls, mode="reduce-overhead")
        return cls  # type: ignore

    @classmethod
    def from_lighting_checkpoint(
        cls,
        checkpoint_path: str | Path,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        compile: bool = False,
        model_keys_starts_with_prefix: str = "model.",
    ) -> "TimesFM_2p5_Model":
        if device is None:
            device = torch.get_default_device()
        if dtype is None:
            dtype = torch.get_default_dtype()

        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_dir():
            raise ValueError(f"{checkpoint_path} is a directory")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{checkpoint_path} does not exist")
        if not checkpoint_path.suffix == ".ckpt":
            raise ValueError(f"{checkpoint_path} is not a .ckpt file")

        ckpt_state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
            mmap=True,
        )
        cls = cls.from_config(
            ckpt_state_dict["config"], dtype=dtype, device=device, compile=compile
        )

        if model_keys_starts_with_prefix != "":
            if model_keys_starts_with_prefix[-1] != ".":
                model_keys_starts_with_prefix += "."

        if model_keys_starts_with_prefix != "":
            model_state_dict = {}
            for key, value in ckpt_state_dict["state_dict"].items():
                if key.startswith(model_keys_starts_with_prefix):
                    new_key = key[len(model_keys_starts_with_prefix) :]
                    model_state_dict[new_key] = value
        else:
            model_state_dict = ckpt_state_dict["state_dict"]

        incompatible_keys = cls.load_state_dict(model_state_dict, strict=False)
        log_incompatible_keys(incompatible_keys)
        return cls

    def save_pretrained(self, save_directory: str | Path) -> None:
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        save_file(state_dict, save_directory / "model.safetensors")
        self.text_encoder_config.save_pretrained(
            save_directory / "text_encoder_config.json"
        )
        return
