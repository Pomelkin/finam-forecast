from transformers.models.modernbert.modeling_modernbert import ModernBertEmbeddings
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer

from .timesfm.layers import ResidualBlock
from .timesfm.layers import Transformer


FSDP_SHARD_MODULES = {
    Transformer,
    ResidualBlock,
    ModernBertEncoderLayer,
    ModernBertEmbeddings,
}
