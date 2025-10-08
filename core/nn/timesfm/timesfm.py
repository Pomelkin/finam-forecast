from pathlib import Path

import orjson
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from safetensors.torch import save_file
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from transformers.models.modernbert import ModernBertConfig
from transformers.models.modernbert import ModernBertModel

from .configs import TimesFM_2p5_200M_Config
from .layers import ResidualBlock
from .layers import Transformer
from .utils import compute_causal_statistics
from .utils import revin
from core.utils import setup_logger


logger = setup_logger(fmt="only_message")


def log_incompatible_keys(
    incompatible_keys: _IncompatibleKeys, model_specific_msg: str = ""
) -> None:
    if incompatible_keys.missing_keys:
        logger.warning(
            f"Missing keys {model_specific_msg}: {incompatible_keys.missing_keys}"
        )
    if incompatible_keys.unexpected_keys:
        logger.warning(
            f"Unexpected keys {model_specific_msg}: {incompatible_keys.unexpected_keys}"
        )
    return


class NewsTimesFM_2p5_Model(nn.Module):
    """TimesFM 2.5 with 200M parameters."""

    def __init__(self, text_encoder_config: ModernBertConfig) -> None:
        super().__init__()
        config = TimesFM_2p5_200M_Config()

        self.text_encoder_config = text_encoder_config
        self.text_encoder = ModernBertModel(text_encoder_config)
        self.text_projection = nn.Linear(
            text_encoder_config.hidden_size,
            config.stacked_transformers.transformer.model_dims,
            bias=False,
        )

        # Names constants.
        self.input_patch_len = config.input_patch_len  # 32
        self.output_patch_len = config.output_patch_len  # 128
        self.m = self.output_patch_len // self.input_patch_len  # 4
        self.num_layers = config.stacked_transformers.num_layers  # 20
        self.num_heads = config.stacked_transformers.transformer.num_heads  # 16
        self.model_dims = config.stacked_transformers.transformer.model_dims  # 1280
        self.num_heads = self.model_dims // self.num_heads  # 80
        self.num_quantiles = len(config.quantiles) + 1  # 10
        self.pred_quantile_index = config.decode_index  # 5

        # Layers.
        self.tokenizer = ResidualBlock(config.tokenizer)
        self.stacked_xf: nn.ModuleList = nn.ModuleList(
            [
                Transformer(config.stacked_transformers.transformer)
                for _ in range(self.num_layers)
            ]
        )
        self.output_projection_point = ResidualBlock(config.output_projection_point)

        self.config = config

        self._init_projections()
        return

    def _init_projections(self) -> None:
        torch.nn.init.normal_(
            self.text_projection.weight,
            mean=0.0,
            std=self.text_encoder_config.initializer_range,
        )
        return

    @property
    def config_dict(self) -> dict:
        config = {"text_encoder": self.text_encoder_config.to_dict()}
        return config

    def forward(
        self,
        inputs_ts: torch.Tensor,
        masks_ts: torch.Tensor,
        inputs_text: torch.Tensor,
    ) -> torch.Tensor:
        B = inputs_ts.shape[0]
        tokenizer_inputs = torch.cat([inputs_ts, masks_ts.to(inputs_ts.dtype)], dim=-1)
        input_embeddings = self.tokenizer(tokenizer_inputs)

        output_embeddings = input_embeddings
        for layer in self.stacked_xf:
            output_embeddings = layer(
                input_embeddings_ts=output_embeddings,
                patch_mask_ts=masks_ts[..., -1],
                input_embedding_text=inputs_text,
            )
        output_ts = self.output_projection_point(output_embeddings)
        output_ts = output_ts.reshape(B, -1, self.output_patch_len, self.num_quantiles)
        return output_ts

    def forecast(
        self,
        inputs_ts: torch.Tensor,  # (B, patch_len, context_len)
        mask_ts: torch.Tensor,  # (B, patch_len, context_len)
        inputs_text: torch.Tensor,  # (B, patch_len, T)
        mask_text: torch.Tensor,  # (B, patch_len, T
        targets: torch.Tensor | None = None,  # (B, patch_len, output_patch_len)
    ) -> dict[str, torch.Tensor]:
        B, patch_len, T = inputs_text.shape

        flattened_inputs_text = inputs_text.reshape(-1, T)
        flattened_mask_text = mask_text.reshape(-1, T)

        text_embeddings = self.text_encoder(
            input_ids=flattened_inputs_text, attention_mask=flattened_mask_text
        ).last_hidden_state

        text_embeddings = text_embeddings.reshape(
            B, patch_len, T, -1
        )  # (B, patch_len, T, hidden)
        mask_text = mask_text.to(text_embeddings.dtype)  # (B, patch_len, T)
        text_embeddings = text_embeddings * mask_text.unsqueeze(
            -1
        )  # (B, patch_len, T, hidden) * (B, patch_len, T, 1)
        mean_pooled = text_embeddings.sum(dim=2) / (
            mask_text.sum(dim=2, keepdim=True) + 1e-8
        )  # (B, patch_len, hidden)

        text_embeddings = self.text_projection(
            mean_pooled
        )  # (B, patch_len, model_dims)

        causal_means, causal_scale = compute_causal_statistics(inputs_ts, mask_ts)
        normalized_inputs = revin(inputs_ts, causal_means, causal_scale, reverse=False)

        normalized_output_ts = self(normalized_inputs, mask_ts, text_embeddings)
        normalized_output_ts = normalized_output_ts[..., self.pred_quantile_index]

        last_causal_mean = causal_means[..., -1].unsqueeze(-1)
        last_causal_scale = causal_scale[..., -1].unsqueeze(-1)

        output_ts = revin(
            normalized_output_ts, last_causal_mean, last_causal_scale, reverse=True
        )

        outputs = {
            "normalized_output": normalized_output_ts,
            "output": output_ts,
        }

        if targets is not None:
            normalized_targets = revin(
                targets, last_causal_mean, last_causal_scale, reverse=False
            )
            outputs["normalized_target"] = normalized_targets
            outputs["target"] = targets
        return outputs

    def compile_(self) -> "NewsTimesFM_2p5_Model":
        return torch.compile(self, mode="reduce-overhead", dynamic=True)  # type: ignore

    @classmethod
    def from_hf(
        cls,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        compile: bool = False,
    ) -> "NewsTimesFM_2p5_Model":
        if device is None:
            device = torch.get_default_device()
        if dtype is None:
            dtype = torch.get_default_dtype()

        ts_encoder = "google/timesfm-2.5-200m-pytorch"
        text_encoder = "deepvk/RuModernBERT-base"

        text_encoder_config_path = hf_hub_download(
            repo_id=text_encoder, filename="config.json", revision="patched-tokenizer"
        )
        text_encoder_config = ModernBertConfig.from_json_file(text_encoder_config_path)

        cls = cls(text_encoder_config=text_encoder_config).to(
            dtype=dtype, device=device
        )

        ts_encoder_weights_path = hf_hub_download(
            repo_id=ts_encoder, filename="model.safetensors"
        )
        state_dict = load_file(ts_encoder_weights_path, device=str(device))
        incompatible_keys = cls.load_state_dict(state_dict, strict=False)
        log_incompatible_keys(
            incompatible_keys, model_specific_msg="for TimesFM_2p5_Model"
        )

        text_encoder_weights_path = hf_hub_download(
            repo_id=text_encoder,
            filename="model.safetensors",
            revision="patched-tokenizer",
        )
        text_encoder_state_dict = load_file(
            text_encoder_weights_path, device=str(device)
        )
        if any("model." in k for k in text_encoder_state_dict.keys()):
            text_encoder_state_dict_temp = {}
            for k, v in text_encoder_state_dict.items():
                if k.startswith("model."):
                    new_key = k[len("model.") :]
                    text_encoder_state_dict_temp[new_key] = v
                else:
                    text_encoder_state_dict_temp[k] = v
            text_encoder_state_dict = text_encoder_state_dict_temp
        incompatible_keys = cls.text_encoder.load_state_dict(
            text_encoder_state_dict, strict=False
        )
        log_incompatible_keys(incompatible_keys, model_specific_msg="for TextEncoder")

        if compile:
            cls = cls.compile_()
        return cls  # type: ignore

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        compile: bool = False,
    ) -> "NewsTimesFM_2p5_Model":
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

        possible_file_names = ["text_encoder_config.json", "config.json"]
        for file_name in possible_file_names:
            if (path / file_name).exists():
                config_file = file_name
                break
        else:
            raise FileNotFoundError(
                f"Neither 'text_encoder_config.json' nor 'config.json' found in {path}"
            )

        text_encoder_config_dict = orjson.loads((path / config_file).read_bytes())
        text_encoder_config = ModernBertConfig.from_dict(text_encoder_config_dict)
        cls = cls(text_encoder_config=text_encoder_config).to(
            dtype=dtype, device=device
        )

        state_dict = load_file(path / "model.safetensors", device=str(device))
        incompatible_keys = cls.load_state_dict(state_dict, strict=False)
        log_incompatible_keys(incompatible_keys)
        if compile:
            cls = cls.compile_()
        return cls  # type: ignore

    @classmethod
    def from_config(
        cls,
        config: dict | ModernBertConfig,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        compile: bool = False,
    ) -> "NewsTimesFM_2p5_Model":
        if isinstance(config, dict):
            if "text_encoder" in config:
                config = config["text_encoder"]
            text_encoder_config = ModernBertConfig.from_dict(config)  # type: ignore
        else:
            text_encoder_config = config

        cls = cls(text_encoder_config=text_encoder_config).to(
            dtype=dtype, device=device
        )
        if compile:
            cls = cls.compile_()
        return cls  # type: ignore

    @classmethod
    def from_lighting_checkpoint(
        cls,
        checkpoint_path: str | Path,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        compile: bool = False,
        model_keys_starts_with_prefix: str = "model.",
    ) -> "NewsTimesFM_2p5_Model":
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
        self.text_encoder_config.save_pretrained(save_directory)
        return
