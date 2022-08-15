from typing import Callable

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from ltp_core.models.components.token import TokenClassifierResult
from ltp_core.models.functional.distill import kd_ce_loss, kd_mse_loss


class TokenLoss(Module):
    def forward(self, result: TokenClassifierResult, labels: Tensor, *args, **kwargs) -> Tensor:
        loss = None

        crf = result.crf
        logits = result.logits
        num_tags = logits.shape[-1]
        attention_mask = result.attention_mask

        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
            # Only keep active parts of the loss
            if crf is not None:
                logits = torch.log_softmax(logits, dim=-1)
                loss = -crf(logits, labels, attention_mask)
            elif attention_mask is not None:
                active_loss = attention_mask.view(-1)
                active_logits = logits.view(-1, num_tags)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, num_tags), labels.view(-1))
        return loss


class SRLLoss(Module):
    def forward(self, result: TokenClassifierResult, labels: Tensor, *args, **kwargs) -> Tensor:
        loss = None

        crf = result.crf
        logits = result.logits
        num_tags = logits.shape[-1]
        attention_mask = result.attention_mask

        # to expand
        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, attention_mask.size(1))
        attention_mask = attention_mask & torch.transpose(attention_mask, -1, -2)
        attention_mask = attention_mask.flatten(end_dim=1)

        index = attention_mask[:, 0]
        attention_mask = attention_mask[index]
        logits = logits.flatten(end_dim=1)[index]
        labels = labels.flatten(end_dim=1)[index]

        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
            # Only keep active parts of the loss
            if crf is not None:
                logits = torch.log_softmax(logits, dim=-1)
                loss = -crf(logits, labels, attention_mask)
            elif attention_mask is not None:
                active_loss = attention_mask.view(-1)
                active_logits = logits.view(-1, num_tags)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, num_tags), labels.view(-1))
        return loss


class TokenDistillLoss(TokenLoss):
    def __init__(self, temperature_scheduler: Callable[[Tensor, Tensor], float]):
        super().__init__()
        self.temperature_scheduler = temperature_scheduler

    def forward(
        self,
        result: TokenClassifierResult,
        labels: Tensor,
        targets: Tensor = None,
        *args,
        **kwargs
    ) -> Tensor:
        loss = super().forward(result, labels, **kwargs)

        if targets is not None:
            crf = result.crf
            logits = result.logits
            num_tags = logits.shape[-1]
            attention_mask = result.attention_mask

            active_loss = attention_mask.view(-1)
            active_logits = logits.view(-1, num_tags)[active_loss]
            active_target = targets.view(-1, num_tags)[active_loss]

            temperature = self.temperature_scheduler(active_logits, active_target)
            distill_loss = kd_ce_loss(active_logits, active_target, temperature)
            loss = loss + distill_loss

            if crf is not None:
                pass

        return loss


class SRLDistillLoss(SRLLoss):
    def __init__(self, temperature_scheduler: Callable[[Tensor, Tensor], float]):
        super().__init__()
        self.temperature_scheduler = temperature_scheduler

    def forward(
        self,
        result: TokenClassifierResult,
        labels: Tensor,
        targets: Tensor = None,
        *args,
        **kwargs
    ) -> Tensor:
        loss = super().forward(result, labels, **kwargs)

        if targets is not None:

            crf = result.crf
            logits = result.logits
            num_tags = logits.shape[-1]
            attention_mask = result.attention_mask

            active_loss = attention_mask.view(-1)
            active_logits = logits.view(-1, num_tags)[active_loss]
            active_target = targets.view(-1, num_tags)[active_loss]

            temperature = self.temperature_scheduler(active_logits, active_target)
            distill_loss = kd_ce_loss(active_logits, active_target, temperature)
            loss = loss + distill_loss

            if crf is None:
                pass
            else:
                pass

        return loss
