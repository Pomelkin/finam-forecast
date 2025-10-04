import math

import torch
import torch.distributed as dist
from torch import nn

from .dist_dot_prod import distributed_2d_dot_product


class ClipLoss(nn.Module):
    def __init__(self, use_distributed_dot_product: bool = True) -> None:
        """
        Initialize the CLIP-style contrastive loss module.

        This sets up:
        - A learnable logit_scale parameter (temperature) initialized to log(1 / 0.07), matching the original CLIP paper default.
        - A per-sample cross-entropy loss (no reduction) used to compute symmetric image-text (or modality A–B) contrastive losses.
        - A flag controlling whether similarity (dot product) computation is performed in a distributed fashion (e.g., across DDP workers) so that the effective batch for the softmax includes samples from all processes.

        Args:
            use_distributed_dot_product (bool, optional):
                If True (default), computes the full similarity matrix across all distributed processes (e.g., via all-gather),
                enabling larger effective batch size and better alignment. If False, similarity is computed only within the local process.

        Returns:
            None
        """
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.use_distributed_dot_product = use_distributed_dot_product
        return

    def forward(
        self,
        text_features: torch.Tensor,
        timeseries_features: torch.Tensor,
        process_group: dist.ProcessGroup | None = None,
    ) -> tuple[torch.Tensor, int]:
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        timeseries_features = timeseries_features / timeseries_features.norm(
            dim=-1, keepdim=True
        )
        if self.use_distributed_dot_product:
            text_to_ts = distributed_2d_dot_product(
                text_features, timeseries_features, group=process_group
            )
        else:
            text_to_ts = text_features @ timeseries_features.t()

        B = text_to_ts.shape[0]

        text_to_ts = self.logit_scale.exp() * text_to_ts
        ts_to_text = text_to_ts.t()
        targets = torch.arange(text_to_ts.size(0), device=text_to_ts.device)

        loss = (self.ce(text_to_ts, targets) + self.ce(ts_to_text, targets)).mean() / 2
        return loss, B
