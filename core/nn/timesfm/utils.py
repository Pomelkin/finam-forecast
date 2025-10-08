import torch

_TOLERANCE = 1e-6


def update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Updates the running stats."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    is_legit = torch.logical_not(mask)
    inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)

    inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
    inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
    inc_mu = inc_mu_numerator / inc_n_safe
    inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

    inc_var_numerator = torch.sum(((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1)
    inc_var = inc_var_numerator / inc_n_safe
    inc_var = torch.where(inc_n == 0, 0.0, inc_var)
    inc_sigma = torch.sqrt(inc_var)

    new_n = n + inc_n
    new_n_safe = torch.where(new_n == 0, 1.0, new_n)

    new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
    new_mu = torch.where(new_n == 0, 0.0, new_mu)

    term1 = n * sigma.pow(2)
    term2 = inc_n * inc_sigma.pow(2)
    term3 = n * (mu - new_mu).pow(2)
    term4 = inc_n * (inc_mu - new_mu).pow(2)

    new_var = (term1 + term2 + term3 + term4) / new_n_safe
    new_var = torch.where(new_n == 0, 0.0, new_var)
    new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

    return new_n, new_mu, new_sigma


def revin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reverse: bool = False,
):
    """Reversible instance normalization."""
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]

    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)


def compute_causal_statistics(
    data: torch.Tensor,
    weights: torch.Tensor,
    minimum_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute causal means at each time step
    input_dtype = data.dtype
    B, T, C = data.shape

    if len(data.shape) > 2:
        data = data.reshape(B, -1)
    if len(weights.shape) > 2:
        weights = weights.reshape(B, -1)
    if weights.dtype != torch.bool:
        weights = weights.bool()

    weights = ~weights
    weighted_data = weights * data
    cum_weights = torch.cumsum(weights, dim=-1)
    cum_values = torch.cumsum(weighted_data, dim=-1)
    denominator = cum_weights.clamp_min(1.0)
    causal_means = cum_values / denominator

    # For Welford’s algorithm, we need to compute the correction term
    # delta using the difference between the current value and the
    # previous running mean.
    shifted_means = torch.zeros_like(causal_means)
    shifted_means[..., 1:] = causal_means[..., :-1]
    delta = data - shifted_means

    # Compute m_2, the second moment accumulator for Welford’s
    # algorithm.
    increment = delta * (data - causal_means) * weights
    m_2 = torch.cumsum(increment, dim=-1)

    # Compute the variance using Bessel’s correction.
    causal_variance = m_2 / torch.clamp(denominator - 1.0, min=1.0)
    causal_scale = torch.sqrt(causal_variance + minimum_scale)

    causal_means = causal_means.reshape(B, T, C)
    causal_scale = causal_scale.reshape(B, T, C)
    return causal_means.to(input_dtype), causal_scale.to(input_dtype)
