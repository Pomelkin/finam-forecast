from pathlib import Path
from typing import Literal
from typing import TypeVar

import orjson
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from transformers.models.modernbert import ModernBertConfig as HFModernBertConfig

TClass = TypeVar("TClass", bound="ConfigMixin")


class ConfigMixin(PydanticBaseModel):
    @classmethod
    def from_json(cls: type[TClass], path: str | Path) -> TClass:
        """Initialize config from a JSON file."""

        if isinstance(path, str):
            path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        elif path.suffix != ".json":
            raise ValueError(f"Config file must be a .json file: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        config = orjson.loads(path.read_bytes())
        return cls.model_validate(config)

    @classmethod
    def from_dict(cls: type[TClass], config: dict) -> TClass:
        return cls.model_validate(config)

    def to_dict(self) -> dict:
        return self.model_dump()

    def save_pretrained(self, path: str | Path) -> None:
        """Save config to a JSON file."""

        if isinstance(path, str):
            path = Path(path)

        match path.suffix:
            case "":
                path = path / "config.json"
            case ".json":
                pass
            case _:
                raise ValueError(f"Config file must be a .json file: {path}")

        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        path.write_bytes(orjson.dumps(self.model_dump(), option=orjson.OPT_INDENT_2))
        return


class ResidualBlockConfig(PydanticBaseModel):
    """Framework-agnostic config for a residual block."""

    input_dims: int
    hidden_dims: int
    output_dims: int
    use_bias: bool
    activation: Literal["relu", "swish", "none"]


class RandomFourierFeaturesConfig(PydanticBaseModel):
    """Framework-agnostic config for random fourier features."""

    input_dims: int
    output_dims: int
    projection_stddev: float
    use_bias: bool


class TransformerConfig(PydanticBaseModel):
    """Framework-agnostic config for a transformer."""

    model_dims: int
    hidden_dims: int
    num_heads: int
    attention_norm: Literal["rms"]
    feedforward_norm: Literal["rms"]
    qk_norm: Literal["rms", "none"]
    use_bias: bool
    use_rotary_position_embeddings: bool
    ff_activation: Literal["relu", "swish", "none"]
    fuse_qkv: bool


class StackedTransformersConfig(PydanticBaseModel):
    """Framework-agnostic config for a stacked transformers."""

    num_layers: int
    transformer: TransformerConfig


class ModernBertConfig(PydanticBaseModel):
    model_config = ConfigDict(extra="allow")
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    norm_eps: float = 1e-5
    norm_bias: bool = False
    pad_token_id: int = 50283
    eos_token_id: int = 50282
    bos_token_id: int = 50281
    cls_token_id: int = 50281
    sep_token_id: int = 50282
    global_rope_theta: float = 160000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    local_rope_theta: float = 10000.0
    embedding_dropout: float = 0.0
    mlp_bias: bool = False
    mlp_dropout: float = 0.0
    decoder_bias: bool = True
    classifier_pooling: Literal["cls", "mean"] = "cls"
    classifier_dropout: float = 0.0
    classifier_bias: bool = False
    classifier_activation: str = "gelu"
    deterministic_flash_attn: bool = False
    sparse_prediction: bool = False
    sparse_pred_ignore_index: int = -100
    reference_compile: str | None = None
    repad_logits_with_grad: bool = False

    def to_hf(self) -> HFModernBertConfig:
        cfg = self.model_dump()
        return HFModernBertConfig.from_dict(cfg)


class TimesFM_2p5_200M_Config(ConfigMixin):
    """Framework-agnostic config of TimesFM 2.5."""

    context_limit: int = 16384
    input_patch_len: int = 32
    output_patch_len: int = 20
    # decode_index: int = 5
    # quantiles: list[float] = dataclasses.field(
    #     default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # )
    text_encoder: ModernBertConfig = ModernBertConfig()
    tokenizer: ResidualBlockConfig = ResidualBlockConfig(
        input_dims=64,
        hidden_dims=1280,
        output_dims=1280,
        use_bias=True,
        activation="swish",
    )
    stacked_transformers: StackedTransformersConfig = StackedTransformersConfig(
        num_layers=20,
        transformer=TransformerConfig(
            model_dims=1280,
            hidden_dims=1280,
            num_heads=16,
            attention_norm="rms",
            feedforward_norm="rms",
            qk_norm="rms",
            use_bias=False,
            use_rotary_position_embeddings=True,
            ff_activation="swish",
            fuse_qkv=True,
        ),
    )
    output_projection_point: ResidualBlockConfig = ResidualBlockConfig(
        input_dims=1280,
        hidden_dims=1280,
        output_dims=20,
        use_bias=False,
        activation="swish",
    )
