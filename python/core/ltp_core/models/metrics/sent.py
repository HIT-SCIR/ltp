from typing import Optional

from torch import Tensor
from torchmetrics import Accuracy

from ltp_core.models.components.sent import SentClassifierResult


class ClsAccuracy(Accuracy):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def update(self, result: SentClassifierResult, labels: Tensor, **kwargs) -> None:
        preds = result.logits.argmax(dim=-1)
        super().update(preds, labels)
