from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from ltp_core.models.components.sent import SentClassifierResult


class ClassificationLoss(Module):
    def forward(self, result: SentClassifierResult, labels: Tensor, **kwargs) -> Tensor:
        logits = result.logits
        num_tags = logits.shape[-1]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_tags), labels.view(-1))
        return loss
