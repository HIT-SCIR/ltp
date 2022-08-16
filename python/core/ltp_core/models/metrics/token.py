from typing import Any, List, Optional, Union

import torch
from torch import Tensor, tensor
from torchmetrics import Accuracy, Metric

from ltp_core.algorithms import get_entities
from ltp_core.models.components.token import TokenClassifierResult


class TokenAccuracy(Accuracy):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def update(self, result: TokenClassifierResult, labels: Tensor, **kwargs) -> None:
        preds = result.logits.argmax(dim=-1)
        attention_mask = result.attention_mask

        preds = preds[attention_mask]
        labels = labels[attention_mask]

        super().update(preds, labels)


class SeqEvalF1(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    correct: Tensor
    gold_total: Tensor
    pred_total: Tensor

    def __init__(self, tags_or_path: Union[str, List[str]], **kwargs: Any):
        super().__init__(**kwargs)
        if isinstance(tags_or_path, str):
            with open(tags_or_path) as f:
                self.labels = [line.strip() for line in f]
        else:
            self.labels = list(tags_or_path)

        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("gold_total", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, result: TokenClassifierResult, labels: Tensor, **kwargs) -> None:
        crf = result.crf
        logits = result.logits
        attention_mask = result.attention_mask

        labels = labels.cpu().numpy()
        labels = [
            [self.labels[tag] for tag, mask in zip(tags, masks) if mask]
            for tags, masks in zip(labels, attention_mask)
        ]

        if crf is None:
            decoded = logits.argmax(dim=-1)
            decoded = decoded.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            decoded = [
                [self.labels[tag] for tag, mask in zip(tags, masks) if mask]
                for tags, masks in zip(decoded, attention_mask)
            ]
        else:
            logits = torch.log_softmax(logits, dim=-1)
            decoded = crf.decode(logits, attention_mask)
            decoded = [[self.labels[tag] for tag in tags] for tags in decoded]

        gold_entities = get_entities(self.concat(labels))
        pred_entities = get_entities(self.concat(decoded))

        self.correct += len(set(gold_entities) & set(pred_entities))
        self.gold_total += len(gold_entities)
        self.pred_total += len(pred_entities)

    def compute(self) -> Any:
        return 2 * self.correct / (self.pred_total + self.gold_total + 1e-6)

    @staticmethod
    def concat(tags):
        new_tags = []
        for seq_tags in tags:
            new_tags.extend(seq_tags)
            new_tags.append("O")
        return new_tags


class SRLEvalF1(SeqEvalF1):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def update(self, result: TokenClassifierResult, labels: Tensor, **kwargs) -> None:
        crf = result.crf
        logits = result.logits
        attention_mask = result.attention_mask

        # to expand
        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, attention_mask.size(1))
        attention_mask = attention_mask & torch.transpose(attention_mask, -1, -2)
        attention_mask = attention_mask.flatten(end_dim=1)

        index = attention_mask[:, 0]
        attention_mask = attention_mask[index]
        logits = logits.flatten(end_dim=1)[index]
        labels = labels.flatten(end_dim=1)[index]

        labels = labels.cpu().numpy()
        labels = [
            [self.labels[tag] for tag, mask in zip(tags, masks) if mask]
            for tags, masks in zip(labels, attention_mask)
        ]

        if crf is None:
            decoded = logits.argmax(dim=-1)
            decoded = decoded.cpu().numpy()
            decoded = [
                [self.labels[tag] for tag, mask in zip(tags, masks) if mask]
                for tags, masks in zip(decoded, attention_mask)
            ]
        else:
            logits = torch.log_softmax(logits, dim=-1)
            decoded = crf.decode(logits, attention_mask)
            decoded = [[self.labels[tag] for tag in tags] for tags in decoded]

        gold_entities = get_entities(self.concat(labels))
        pred_entities = get_entities(self.concat(decoded))

        self.correct += len(set(gold_entities) & set(pred_entities))
        self.gold_total += len(gold_entities)
        self.pred_total += len(pred_entities)
