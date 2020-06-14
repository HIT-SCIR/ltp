#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch.nn.functional as F
import torch
from typing import List
from . import Loss


class KdMseLoss(Loss, alias='kd_mse_loss'):
    def forward(self, logits_S, logits_T, temperature=1):
        '''
        Calculate the mse loss between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        loss = F.mse_loss(beta_logits_S, beta_logits_T)
        return loss


class KdCeLoss(Loss, alias='kd_ce_loss'):
    def forward(self, logits_S, logits_T, temperature=1):
        '''
        Calculate the cross entropy between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        p_T = F.softmax(beta_logits_T, dim=-1)
        loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
        return loss


class AttMseLoss(Loss, alias='att_mse_loss'):
    def forward(self, attention_S, attention_T, mask=None):
        '''
        Calculate the mse loss between attention_S and attention_T.

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        '''
        if mask is None:
            attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
            attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
            loss = F.mse_loss(attention_S_select, attention_T_select)
        else:
            mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
            valid_count = torch.pow(mask.sum(dim=2), 2).sum()
            loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
                2)).sum() / valid_count
        return loss


class AttMseSumLoss(Loss, alias='att_mse_sum_loss'):
    def forward(self, attention_S, attention_T, mask=None):
        '''
        Calculate the mse loss between attention_S and attention_T, the dim of num_heads is summed

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        '''
        if len(attention_S.size()) == 4:
            attention_T = attention_T.sum(dim=1)
            attention_S = attention_S.sum(dim=1)
        if mask is None:
            attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
            attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
            loss = F.mse_loss(attention_S_select, attention_T_select)
        else:
            mask = mask.to(attention_S)
            valid_count = torch.pow(mask.sum(dim=1), 2).sum()
            loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
                1)).sum() / valid_count
        return loss


class AttCeLoss(Loss, alias='att_ce_loss'):
    def forward(self, attention_S, attention_T, mask=None):
        '''
        Calculate the cross entropy  between attention_S and attention_T.

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        '''
        probs_T = F.softmax(attention_T, dim=-1)
        if mask is None:
            probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
            loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
        else:
            mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
            loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(2)).sum(
                dim=-1) * mask).sum() / mask.sum()
        return loss


class AttCeMeanLoss(Loss, alias='att_ce_mean_loss'):
    def forward(self, attention_S, attention_T, mask=None):
        '''
        Calculate the cross entropy  between attention_S and attention_T, the dim of num_heads is averaged

        :param logits_S: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param logits_T: Tensor of shape  (batch_size, num_heads, length, length) or (batch_size, length, length)
        :param mask:     Tensor of shape  (batch_size, length)
        '''
        if len(attention_S.size()) == 4:
            attention_S = attention_S.mean(dim=1)  # (bs, len, len)
            attention_T = attention_T.mean(dim=1)
        probs_T = F.softmax(attention_T, dim=-1)
        if mask is None:
            probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
            loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
        else:
            mask = mask.to(attention_S)
            loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(1)).sum(
                dim=-1) * mask).sum() / mask.sum()
        return loss


class HidMseLoss(Loss, alias='hid_mse_loss'):
    def forward(self, state_S, state_T, mask=None):
        '''
        Calculate the mse loss between state_S and state_T, state is the hidden state of the model

        :param state_S: Tensor of shape  (batch_size, length, hidden_size)
        :param state_T: Tensor of shape  (batch_size, length, hidden_size)
        :param mask:    Tensor of shape  (batch_size, length)
        '''
        if mask is None:
            loss = F.mse_loss(state_S, state_T)
        else:
            mask = mask.to(state_S)
            valid_count = mask.sum() * state_S.size(-1)
            loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
        return loss


class CosLoss(Loss, alias='cos_loss'):
    def forward(self, state_S, state_T, mask=None):
        '''
        This is the loss used in DistilBERT

        :param state_S: Tensor of shape  (batch_size, length, hidden_size)
        :param state_T: Tensor of shape  (batch_size, length, hidden_size)
        :param mask:    Tensor of shape  (batch_size, length)
        '''
        if mask is None:
            state_S = state_S.view(-1, state_S.size(-1))
            state_T = state_T.view(-1, state_T.size(-1))
        else:
            mask = mask.to(state_S).unsqueeze(-1).expand_as(state_S).to(torch.uint8)  # (bs,len,dim)
            state_S = torch.masked_select(state_S, mask).view(-1, mask.size(-1))  # (bs * select, dim)
            state_T = torch.masked_select(state_T, mask).view(-1, mask.size(-1))  # (bs * select, dim)

        target = state_S.new(state_S.size(0)).fill_(1)
        loss = F.cosine_embedding_loss(state_S, state_T, target, reduction='mean')
        return loss


class PkdLoss(Loss, alias='pkd_loss'):
    def forward(self, state_S, state_T, mask=None):
        '''
        This is the loss used in BERT-PKD

        :param state_S: Tensor of shape  (batch_size, length, hidden_size)
        :param state_T: Tensor of shape  (batch_size, length, hidden_size)
        '''

        cls_T = state_T[:, 0]  # (batch_size, hidden_dim)
        cls_S = state_S[:, 0]  # (batch_size, hidden_dim)
        normed_cls_T = cls_T / torch.norm(cls_T, dim=1, keepdim=True)
        normed_cls_S = cls_S / torch.norm(cls_S, dim=1, keepdim=True)
        loss = (normed_cls_S - normed_cls_T).pow(2).sum(dim=-1).mean()
        return loss


class FspLoss(Loss, alias='fsp_loss'):
    def forward(state_S: List[torch.Tensor], state_T: List[torch.Tensor], mask=None):
        '''
        Variant of FSP loss from A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning

        :param state_S: list of two tensors, each tensor is of shape  (batch_size, length, hidden_size)
        :param state_T: list of two tensors, each tensor is of shape  (batch_size, length, hidden_size)
        :param mask:    Tensor of shape  (batch_size, length)
        '''
        if mask is None:
            state_S_0 = state_S[0]  # (batch_size , length, hidden_dim)
            state_S_1 = state_S[1]  # (batch_size,  length, hidden_dim)
            state_T_0 = state_T[0]
            state_T_1 = state_T[1]
            gram_S = torch.bmm(state_S_0.transpose(1, 2), state_S_1) / state_S_1.size(
                1)  # (batch_size, hidden_dim, hidden_dim)
            gram_T = torch.bmm(state_T_0.transpose(1, 2), state_T_1) / state_T_1.size(1)
        else:
            mask = mask.to(state_S[0]).unsqueeze(-1)
            lengths = mask.sum(dim=1, keepdim=True)
            state_S_0 = state_S[0] * mask
            state_S_1 = state_S[1] * mask
            state_T_0 = state_T[0] * mask
            state_T_1 = state_T[1] * mask
            gram_S = torch.bmm(state_S_0.transpose(1, 2), state_S_1) / lengths
            gram_T = torch.bmm(state_T_0.transpose(1, 2), state_T_1) / lengths
        loss = F.mse_loss(gram_S, gram_T)
        return loss


class MmdLoss(Loss, alias='mmd_loss'):
    def forward(state_S: List[torch.Tensor], state_T: List[torch.Tensor], mask=None):
        '''
        Variant of  NST loss and its from Like What You Like: Knowledge Distill via Neuron Selectivity Transfer

        :param state_S: list of two tensors, each tensor is of shape  (batch_size, length, hidden_size)
        :param state_T: list of two tensors, each tensor is of shape  (batch_size, length, hidden_size)
        :param mask:    Tensor of shape  (batch_size, length)
        '''

        state_S_0 = state_S[0]  # (batch_size , length, hidden_dim_S)
        state_S_1 = state_S[1]  # (batch_size , length, hidden_dim_S)
        state_T_0 = state_T[0]  # (batch_size , length, hidden_dim_T)
        state_T_1 = state_T[1]  # (batch_size , length, hidden_dim_T)
        if mask is None:
            gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(2)  # (batch_size, length, length)
            gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(2)
            loss = F.mse_loss(gram_S, gram_T)
        else:
            mask = mask.to(state_S[0])
            valid_count = torch.pow(mask.sum(dim=1), 2).sum()
            gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(1)  # (batch_size, length, length)
            gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(1)
            loss = (F.mse_loss(gram_S, gram_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(
                1)).sum() / valid_count
        return loss
