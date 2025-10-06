from typing import Any

from torch import nn


def create_params_groups(
    model: nn.Module,
    weight_decay: float,
    lr: float,
) -> list[dict]:
    """Create optimizer parameter groups for a PyTorch model with fine-grained weight decay control."""
    param_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue
        param_group = {"params": param, "lr": lr}

        if (
            ("norm" in name)
            or ("bias" in name)
            or ("embedding" in name)
            or ("tokenizer" in name)
            or ("output_projection_point" in name)
            or ("ln" in name)
            or ("scale" in name)
        ):
            param_group["weight_decay"] = 0.0
        else:
            param_group["weight_decay"] = weight_decay
        param_groups.append(param_group)

    fused_param_groups = _fuse_groups(param_groups)
    return fused_param_groups


def _fuse_groups(param_groups: list[dict]) -> list[dict]:
    fuse_dict: dict[str, dict[str, Any]] = {}
    for group in param_groups:
        group_key = ""
        for key, value in group.items():
            if key != "params":
                group_key += f"_{key}:{value}"

        if group_key not in fuse_dict:
            fuse_dict[group_key] = {"params": []}
            for k, v in group.items():
                if k != "params":
                    fuse_dict[group_key][k] = v
        fuse_dict[group_key]["params"].append(group["params"])
    return list(fuse_dict.values())
