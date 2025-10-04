from typing import Any
from typing import cast

import torch
import torch.distributed as dist


class Distributed2dDotProductFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, y: torch.Tensor, group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        if x.shape[-1] != y.shape[-1]:
            raise ValueError("The last dimension of x and y must be the same.")
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("x and y must be 2D tensors.")

        x = x.contiguous()
        y = y.contiguous()

        x_B, C = x.shape
        y_B, _ = y.shape
        if dist.is_initialized():
            world_size = dist.get_world_size(group)
            rank = dist.get_rank(group)

            x_world = [torch.zeros_like(x) for _ in range(world_size)]
            dist.all_gather(x_world, x, group)

            y_world = [torch.zeros_like(y) for _ in range(world_size)]
            dist.all_gather(y_world, y, group)

            x_offset = x_B * rank
            y_offset = y_B * rank

            x = torch.cat(x_world, dim=0)
            y = torch.cat(y_world, dim=0)
        else:
            world_size = 1
            x_offset = 0
            y_offset = 0

        res = x @ y.T

        ctx.save_for_backward(x, y)
        ctx.x_offset = x_offset
        ctx.y_offset = y_offset
        ctx.world_size = world_size
        return res

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> Any:
        x, y = ctx.saved_tensors
        x_offset = ctx.x_offset
        y_offset = ctx.y_offset
        x_B = x.shape[0] // ctx.world_size
        y_B = y.shape[0] // ctx.world_size

        g = grad_outputs[0]

        dx = g @ y
        dy = g.T @ x

        dx = dx[x_offset : x_offset + x_B]
        dy = dy[y_offset : y_offset + y_B]
        return dx, dy, None


def distributed_2d_dot_product(
    x: torch.Tensor, y: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    out = Distributed2dDotProductFunction.apply(x, y, group)
    out = cast(torch.Tensor, out)
    return out
