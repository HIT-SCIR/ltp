import torch
import torch.nn.functional as F


def flsw_temperature_scheduler_builder(beta=1, gamma=2, base_temperature=8, eps=1e-4, *args):
    """adapted from arXiv:1911.07471."""

    def flsw_temperature_scheduler(logits_S, logits_T):
        v = logits_S.detach()
        t = logits_T.detach()
        with torch.no_grad():
            v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
            t = t / (torch.norm(t, dim=-1, keepdim=True) + eps)
            w = torch.pow((1 - (v * t).sum(dim=-1)), gamma)
            tau = base_temperature + (w.mean() - w) * beta
        return tau

    return flsw_temperature_scheduler


def kd_ce_loss(logits_S, logits_T, temperature=1):
    """Calculate the cross entropy between logits_S and logits_T.

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    """
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1))
    return (temperature * temperature * loss).sum(dim=-1).mean()


def kd_mse_loss(logits_S, logits_T, temperature=1):
    """Calculate the mse loss between logits_S and logits_T.

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    """
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    loss = F.mse_loss(beta_logits_S, beta_logits_T, reduction="none")
    return (temperature * temperature * loss).mean()
