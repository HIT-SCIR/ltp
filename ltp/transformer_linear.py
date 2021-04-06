#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import namedtuple
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from ltp.nn import BaseModule, MLP, CRF

TokenClassifierResult = namedtuple('TokenClassifierResult', ['loss', 'logits', 'decoded', 'labels'])


class LinearClassifier(nn.Linear):
    def __init__(self, input_size, num_labels):
        super().__init__(input_size, num_labels)

    def forward(self, input, attention_mask=None, word_index=None, word_attention_mask=None,
                labels=None, is_processed=False) -> TokenClassifierResult:
        if not is_processed:
            input = input[:, 1:-1, :]
            if word_attention_mask is None and attention_mask is not None:
                word_attention_mask = attention_mask[:, 2:] == 1
            if word_index is not None:
                input = torch.gather(input, dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, input.size(-1)))

        logits = super(LinearClassifier, self).forward(input)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if word_attention_mask is not None:
                active_loss = word_attention_mask.view(-1)
                active_logits = logits.view(-1, self.out_features)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.out_features), labels.view(-1))

        return TokenClassifierResult(loss=loss, logits=logits, decoded=None, labels=None)


class MLPClassfier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_labels, dropout,
                 use_cls=False, use_sep=False, use_crf=False, crf_reduction='sum'):
        super().__init__()
        self.use_cls = use_cls
        self.use_sep = use_sep
        self.linear = MLP([input_size, *hidden_sizes, num_labels], dropout=dropout)
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
            self.crf_reduction = crf_reduction
        else:
            self.crf = None

    def forward(self, input, attention_mask=None, word_index=None, word_attention_mask=None,
                labels=None, is_processed=False) -> TokenClassifierResult:
        if not is_processed:
            if not self.use_cls:
                input = input[:, 1:, :]
            if not self.use_cls:
                input = input[:, :-1, :]

            if word_attention_mask is None:
                assert word_index is None
                bias = int(not self.use_cls) + int(not self.use_sep)
                word_attention_mask = attention_mask[:, bias:] == 1

            if word_index is not None:
                input = torch.gather(input, dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, input.size(-1)))

        logits = self.linear.forward(input)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if word_attention_mask is not None and self.crf is not None:
                logits = F.log_softmax(logits, dim=-1)
                loss = - self.crf.forward(logits, labels, word_attention_mask, reduction=self.crf_reduction)
            elif word_attention_mask is not None:
                active_loss = word_attention_mask.view(-1)
                active_logits = logits.view(-1, self.classifier.out_features)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        decoded = None
        if not self.training and self.crf is not None:
            decoded = self.crf.decode(emissions=logits, mask=word_attention_mask)
            if self.use_cls:
                decoded = [sent[1:] for sent in decoded]
                labels = labels[:, 1:]
            if self.use_sep:
                decoded = [sent[:-1] for sent in decoded]
                labels = labels[:, :-1]

        return TokenClassifierResult(loss=loss, logits=logits, decoded=decoded, labels=labels)


class TransformerLinear(BaseModule):
    def __init__(self, hparams, config=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.transformer = AutoModel.from_pretrained(self.hparams.transformer)
        else:
            self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(self.hparams.dropout)
        transformer_hidden_size = self.transformer.config.hidden_size

        if self.hparams.use_mlp:
            self.classifier = MLPClassfier(
                input_size=transformer_hidden_size,
                hidden_sizes=self.hparams.hidden_sizes,
                num_labels=self.hparams.num_labels,
                dropout=self.hparams.dropout,
                use_cls=self.hparams.use_cls,
                use_crf=self.hparams.use_crf,
                use_sep=self.hparams.use_sep,
                crf_reduction=self.hparams.crf_reduction,
            )
        else:
            self.classifier = LinearClassifier(
                input_size=transformer_hidden_size,
                num_labels=self.hparams.num_labels
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--hidden_sizes', nargs='+', type=int)
        parser.add_argument('--num_labels', type=int)
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--use_crf', action='store_true')
        parser.add_argument('--use_cls', action='store_true')
        parser.add_argument('--use_sep', action='store_true')
        parser.add_argument('--crf_reduction', type=str, default='sum')
        return parser

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            word_index=None,
            word_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None
    ) -> TokenClassifierResult:
        hidden_states = self.transformer(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        sequence_output = hidden_states[0]
        sequence_output = self.dropout(sequence_output)

        return self.classifier(
            input=sequence_output,
            attention_mask=attention_mask,
            word_attention_mask=word_attention_mask,
            labels=labels
        )
