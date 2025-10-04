from torch.nn import Parameter
from torch.optim import Optimizer


def gather_params_from_optim(
    optimizers: list[Optimizer] | Optimizer,
) -> list[Parameter]:
    """
    Collect (and optionally deduplicate) torch.nn.Parameter objects from one or more torch.optim.Optimizer instances.
    """
    params = []
    if isinstance(optimizers, Optimizer):
        for pg in optimizers.param_groups:
            params.extend(pg["params"])
    else:
        uniq_id = set()
        for optimizer in optimizers:
            for pg in optimizer.param_groups:
                for param in pg.get("params", []):
                    if id(param) not in uniq_id:
                        uniq_id.add(id(param))
                        params.append(param)
    return params
