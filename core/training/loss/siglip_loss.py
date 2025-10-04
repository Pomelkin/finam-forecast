import math

import torch
import torch.distributed as dist
from torch import nn

from .dist_dot_prod import distributed_2d_dot_product


class SigLipLoss(nn.Module):
    def __init__(self, use_distributed_dot_product: bool = True) -> None:
        """
        Initialize the SigLIP loss module.

        Args:
            use_distributed_dot_product (bool, optional):
                If True, enables a distributed (e.g., multi-GPU) computation path for
                the pairwise dot products / similarities before applying the loss.
                Set to False to restrict computations to the local process only.
        """
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-10.0))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
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
            features = distributed_2d_dot_product(
                text_features, timeseries_features, group=process_group
            )
        else:
            features = text_features @ timeseries_features.t()
        features = self.logit_scale.exp() * features + self.bias

        B = features.shape[0]

        targets = torch.eye(features.size(0), device=features.device)
        loss = self.bce(features, targets)
        loss = loss.sum(dim=-1).mean()
        return loss, B
