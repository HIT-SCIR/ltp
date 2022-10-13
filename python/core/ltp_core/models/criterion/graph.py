import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module

from ltp_core.models.components.graph import GraphResult


class DEPLoss(Module):
    def __init__(self, loss_interpolation=0.4):
        super().__init__()
        self.loss_interpolation = loss_interpolation

    def forward(self, result: GraphResult, head: Tensor, labels: Tensor, **kwargs):
        s_arc = result.arc_logits
        s_rel = result.rel_logits
        attention_mask = result.attention_mask

        arc_loss = CrossEntropyLoss()
        rel_loss = CrossEntropyLoss()

        # ignore the first token of each sentence
        s_arc = s_arc[:, 1:, :]
        s_rel = s_rel[:, 1:, :]

        # Only keep active parts of the loss
        active_heads = head[attention_mask]
        active_labels = labels[attention_mask]
        s_arc, s_rel = s_arc[attention_mask], s_rel[attention_mask]

        s_rel = s_rel[torch.arange(len(active_heads)), active_heads]

        arc_loss = arc_loss(s_arc, active_heads)
        rel_loss = rel_loss(s_rel, active_labels)
        loss = 2 * ((1 - self.loss_interpolation) * arc_loss + self.loss_interpolation * rel_loss)

        return loss


class SDPLoss(Module):
    def __init__(self, loss_interpolation=0.4):
        super().__init__()
        self.loss_interpolation = loss_interpolation

    def forward(self, result: GraphResult, head: Tensor, labels: Tensor, **kwargs):
        s_arc = result.arc_logits
        s_rel = result.rel_logits
        attention_mask = result.attention_mask

        head_loss = BCEWithLogitsLoss()
        rel_loss = CrossEntropyLoss()

        # ignore the first token of each sentence
        s_arc = s_arc[:, 1:, :]
        s_rel = s_rel[:, 1:, :]

        # attention mask
        attention_mask = attention_mask.unsqueeze(-1).expand_as(s_arc)

        arc_loss = head_loss(s_arc[attention_mask], head[attention_mask].float())
        rel_loss = rel_loss(s_rel[head > 0], labels[head > 0])

        loss = 2 * ((1 - self.loss_interpolation) * arc_loss + self.loss_interpolation * rel_loss)

        return loss


class DEPDistillLoss(DEPLoss):
    def forward(self, result: GraphResult, head: Tensor, labels: Tensor, **kwargs):
        return super().forward(result, labels, **kwargs)


class SDPDistillLoss(SDPLoss):
    def forward(self, result: GraphResult, head: Tensor, labels: Tensor, **kwargs):
        return super().forward(result, labels, **kwargs)
