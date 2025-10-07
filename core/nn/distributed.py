from transformers.models.modernbert.modeling_modernbert import ModernBertEmbeddings
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer

from .timesfm.layers import ResidualBlock
from .timesfm.layers import Transformer


FDSP_NO_SPLIT_MODULES = {
    Transformer,
    ResidualBlock,
    ModernBertEncoderLayer,
    ModernBertEmbeddings,
}
