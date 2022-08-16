from typing import Any, Optional

import torch
from torch import Tensor, tensor
from torchmetrics import Metric

from ltp_core.algorithms import eisner
from ltp_core.models.components.graph import GraphResult


class DEPLas(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, result: GraphResult, head: Tensor, labels: Tensor):
        s_arc = result.arc_logits.detach()
        s_rel = result.rel_logits.detach()
        attention_mask = result.attention_mask

        # mask padding 的部分
        word_cls_mask = torch.cat([attention_mask[:, :1], attention_mask], dim=1)
        activate_word_mask = word_cls_mask.unsqueeze(-1).expand_as(s_arc)
        activate_word_mask = activate_word_mask & activate_word_mask.transpose(-1, -2)
        s_arc.masked_fill_(~activate_word_mask, float("-inf"))

        # mask root 和 对角线部分
        s_arc[:, 0, 1:] = float("-inf")
        s_arc.diagonal(0, 1, 2).fill_(float("-inf"))
        arcs = eisner(s_arc, word_cls_mask, True)

        rels = torch.argmax(s_rel[:, 1:], dim=-1)
        rels = rels.gather(-1, arcs.unsqueeze(-1)).squeeze(-1)

        # todo: UAS, now only LAS
        arc_correct = arcs == head
        rel_correct = rels == labels
        self.correct += (arc_correct & rel_correct)[attention_mask].sum().item()
        self.total += torch.sum(attention_mask).item()

    def compute(self) -> Any:
        return self.correct / (self.total + 1e-6)


class SDPLas(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_total", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("gold_total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, result: GraphResult, head: Tensor, labels: Tensor):
        s_arc = result.arc_logits.detach()
        s_rel = result.rel_logits.detach()
        attention_mask = result.attention_mask

        # mask padding 的部分
        activate_word_mask = torch.cat([attention_mask[:, :1], attention_mask], dim=1)
        activate_word_mask = activate_word_mask.unsqueeze(-1).expand_as(s_arc)
        activate_word_mask = activate_word_mask & activate_word_mask.transpose(-1, -2)
        s_arc = s_arc.masked_fill(~activate_word_mask, float("-inf"))

        # mask root 和 对角线部分
        s_arc[:, 0, 1:] = float("-inf")
        s_arc.diagonal(0, 1, 2).fill_(float("-inf"))

        arcs = s_arc[:, 1:, :] > 0
        rels = torch.argmax(s_rel[:, 1:, :], dim=-1)

        pred_entities = self.get_graph_entities(arcs, rels, flatten=True)
        gold_entities = self.get_graph_entities(head, labels, flatten=True)

        self.correct += len(set(gold_entities) & set(pred_entities))
        self.gold_total += len(gold_entities)
        self.pred_total += len(pred_entities)

    def compute(self) -> Any:
        return 2 * self.correct / (self.pred_total + self.gold_total + 1e-6)

    @staticmethod
    def get_graph_entities(arcs, rels, flatten=False):
        sequence_num = rels.shape[0]
        arcs = torch.nonzero(arcs, as_tuple=False).cpu().detach().numpy().tolist()
        rels = rels.cpu().detach().numpy()

        if flatten:
            res = []
        else:
            res = [[] for _ in range(sequence_num)]
        for idx, arc_s, arc_e in arcs:
            label = rels[idx, arc_s, arc_e]
            if flatten:
                res.append((idx, arc_s, arc_e, label))
            else:
                res[idx].append((arc_s, arc_e, label))

        return res
