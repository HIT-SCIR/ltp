#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from . import Loss
from .kd_loss import KdMseLoss, KdCeLoss
import torch
import torch.nn.functional as F
from ltp.utils import length_to_mask, select_logits_with_mask


class SegLoss(Loss, alias='seg'):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SegLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        mask = length_to_mask(targets['text_length'])
        target = targets['word_idn']
        loss = F.cross_entropy(inputs[mask], target[mask], reduction=self.reduction)
        return loss

    def distill(self, inputs, targets, temperature_calc, distill_loss, gold=None):
        mask = length_to_mask(gold['text_length'])
        logits = inputs[mask]
        logits_T = targets[mask]
        temperature = temperature_calc(logits, logits_T)
        return distill_loss(logits, logits_T, temperature)


class DepLoss(Loss, alias='dep'):
    def __init__(self, size_average=None, ignore_index=-1, loss_interpolation=0.4,
                 reduce=None, reduction='mean'):
        super(DepLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.loss_interpolation = loss_interpolation
        self.kd_ce_loss = KdCeLoss()

    def forward(self, inputs, targets):
        arcs, rels = targets
        arc_scores, rel_scores, seq_lens = inputs
        mask = length_to_mask(seq_lens + 1)
        mask[:, 0] = False  # ignore the first token of each sentence

        arc_scores, rel_scores = arc_scores[mask], rel_scores[mask]

        # for taget not bos
        mask = torch.narrow(mask, dim=-1, start=1, length=mask.size(1) - 1)
        arcs, rels = arcs[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]

        arc_loss = F.cross_entropy(
            arc_scores, arcs, weight=None,
            ignore_index=self.ignore_index, reduction=self.reduction
        )
        rel_loss = F.cross_entropy(
            rel_scores, rels, weight=None,
            ignore_index=self.ignore_index, reduction=self.reduction
        )
        loss = 2 * ((1 - self.loss_interpolation) * arc_loss + self.loss_interpolation * rel_loss)
        return loss

    def distill(self, inputs, targets, temperature_calc, distill_loss, gold=None):
        arc_scores, rel_scores, seq_lens = inputs
        arc_scores_T, rel_scores_T, _ = targets

        mask = length_to_mask(seq_lens + 1)
        mask[:, 0] = False  # ignore the first token of each sentence

        arc_logits = select_logits_with_mask(arc_scores, mask)
        arc_logits_T = select_logits_with_mask(arc_scores_T, mask)
        arc_temperature = temperature_calc(arc_logits, arc_logits_T)

        rel_logits = select_logits_with_mask(rel_scores, mask)
        rel_logits_T = select_logits_with_mask(rel_scores_T, mask)
        rel_temperature = temperature_calc(rel_logits, rel_logits_T)

        loss = 2 * ((1 - self.loss_interpolation) * self.kd_ce_loss(arc_logits, arc_logits_T, arc_temperature)
                    + self.loss_interpolation * self.kd_ce_loss(rel_logits, rel_logits_T, rel_temperature))

        return loss


class SDPLoss(Loss, alias='sdp'):
    def __init__(self, weight=None, size_average=None, ignore_index=-1, loss_interpolation=0.4,
                 reduce=None, reduction='mean'):
        super(SDPLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.loss_interpolation = loss_interpolation
        self.kd_mse_loss = KdMseLoss()
        self.kd_ce_loss = KdCeLoss()

    def forward(self, inputs, targets):
        arcs, rels = targets
        arc_scores, rel_scores, seq_lens = inputs
        mask = length_to_mask(seq_lens + 1, dtype=torch.float)
        mask[:, 0] = 0  # ignore the first token of each sentence

        mask = mask.unsqueeze(-1)
        mask = mask.expand_as(arcs)

        arc_loss = F.binary_cross_entropy_with_logits(
            arc_scores, arcs, weight=mask, reduction=self.reduction
        )

        num_tags = rel_scores.shape[-1]
        rel_loss = F.cross_entropy(
            rel_scores.contiguous().view((-1, num_tags)), rels.contiguous().view(-1), weight=self.weight,
            ignore_index=self.ignore_index, reduction=self.reduction
        )
        loss = 2 * ((1 - self.loss_interpolation) * arc_loss + self.loss_interpolation * rel_loss)
        return loss

    def distill(self, inputs, targets, temperature_calc, distill_loss, gold=None):
        arc_scores, rel_scores, seq_lens = inputs
        arc_scores_T, rel_scores_T, _ = targets

        mask = length_to_mask(seq_lens + 1)
        mask[:, 0] = False

        arc_mask = mask.unsqueeze(-1).expand_as(arc_scores)
        arc_logits = torch.sigmoid(arc_scores)[arc_mask]
        arc_logits_T = torch.sigmoid(arc_scores_T)[arc_mask]
        arc_temperature = temperature_calc(arc_logits, arc_logits_T)

        rel_logits = select_logits_with_mask(rel_scores, mask)
        rel_logits_T = select_logits_with_mask(rel_scores_T, mask)
        rel_temperature = temperature_calc(rel_logits, rel_logits_T)

        loss = 2 * ((1 - self.loss_interpolation) * F.mse_loss(arc_logits / arc_temperature,
                                                               arc_logits_T / arc_temperature)
                    + self.loss_interpolation * self.kd_ce_loss(rel_logits, rel_logits_T, rel_temperature))
        return loss


class BiaffineCRF(Loss, alias='biaffine_crf'):
    def __init__(self, reduction='sum', cross_entropy=False):
        super(BiaffineCRF, self).__init__()
        self.reduction = reduction
        self.cross_entropy = cross_entropy
        self.kd_mse_loss = KdMseLoss()

    def forward(self, inputs, targets):
        emissions, seq_lens, crf = inputs
        rel_gold, rel_gold_set = targets

        mask = length_to_mask(seq_lens)
        mask = mask.unsqueeze_(-1).expand_as(rel_gold)
        mask = mask & mask.transpose(-1, -2)
        mask = mask.flatten(end_dim=1)
        index = mask[:, 0]

        mask = mask[index]
        emissions = emissions.flatten(end_dim=1)[index]
        emissions = F.log_softmax(emissions, dim=-1)
        rel_gold = rel_gold.flatten(end_dim=1)[index]

        if self.cross_entropy:
            cross_entropy = F.cross_entropy(emissions[mask], rel_gold[mask])
            crf_loss = crf.forward(emissions=emissions, tags=rel_gold, mask=mask, reduction=self.reduction)
            return cross_entropy - crf_loss
        else:
            return -crf.forward(emissions=emissions, tags=rel_gold, mask=mask, reduction=self.reduction)

    def distill(self, inputs, targets, temperature_calc, distill_loss, gold=None):
        emissions, seq_lens, crf = inputs
        emissions_T, _, crf_T = targets

        mask = length_to_mask(seq_lens)
        mask = mask.unsqueeze_(-1).expand(-1, -1, mask.size(1))
        mask = mask & mask.transpose(-1, -2)

        logits_loss = F.mse_loss(emissions[mask], emissions_T[mask])
        crf_loss = F.mse_loss(crf.transitions, crf_T.transitions) + \
                   F.mse_loss(crf.start_transitions, crf_T.start_transitions) + \
                   F.mse_loss(crf.end_transitions, crf_T.end_transitions)

        return logits_loss + crf_loss
