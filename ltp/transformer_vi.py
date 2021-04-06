#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from packaging import version
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from ltp.nn import (
    SharedDropout, MLP, LSTM, Bilinear, BaseModule, Triaffine,
    MFVISemanticDependency, LBPSemanticDependency,
)
from ltp.transformer_biaffine import GraphResult

torch_version = version.parse(torch.__version__)
pack_padded_sequence_cpu_version = version.parse('1.7.0')


class ViClassifier(nn.Module):
    def __init__(self, input_size, label_num, dropout,
                 lstm_hidden_size=600, lstm_num_layers=3, bin_hidden_size=150,
                 arc_hidden_size=600, rel_hidden_size=600, loss_interpolation=0.4, inference='mfvi', max_iter=3):
        super().__init__()
        self.label_num = label_num
        self.loss_interpolation = loss_interpolation

        if lstm_num_layers > 0:
            self.lstm = LSTM(
                input_size=input_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                bidirectional=True,
                dropout=dropout
            )
            self.lstm_dropout = SharedDropout(p=dropout)
            hidden_size = lstm_hidden_size * 2
        else:
            self.lstm = None
            hidden_size = input_size

        self.mlp_bin_d = MLP([hidden_size, bin_hidden_size], output_dropout=dropout)
        self.mlp_bin_h = MLP([hidden_size, bin_hidden_size], output_dropout=dropout)
        self.mlp_bin_g = MLP([hidden_size, bin_hidden_size], output_dropout=dropout)
        self.mlp_arc_h = MLP([hidden_size, arc_hidden_size], output_dropout=dropout)
        self.mlp_arc_d = MLP([hidden_size, arc_hidden_size], output_dropout=dropout)
        self.mlp_rel_h = MLP([hidden_size, rel_hidden_size], output_dropout=dropout)
        self.mlp_rel_d = MLP([hidden_size, rel_hidden_size], output_dropout=dropout)

        self.sib_attn = Triaffine(bin_hidden_size, bias_x=True, bias_y=True)
        self.cop_attn = Triaffine(bin_hidden_size, bias_x=True, bias_y=True)
        self.grd_attn = Triaffine(bin_hidden_size, bias_x=True, bias_y=True)
        self.arc_atten = Bilinear(arc_hidden_size, arc_hidden_size, 1, bias_x=True, bias_y=True, expand=True)
        self.rel_atten = Bilinear(rel_hidden_size, rel_hidden_size, label_num, bias_x=True, bias_y=True, expand=True)

        self.vi = (MFVISemanticDependency if inference == 'mfvi' else LBPSemanticDependency)(max_iter)

    def forward(self, input, attention_mask=None, word_index=None, word_attention_mask=None, head=None, labels=None,
                is_processed=False):
        if not is_processed:
            assert word_attention_mask is not None
            input = input[:, :-1, :]
            if word_index is not None:
                input = torch.cat([input[:, :1, :], torch.gather(
                    input[:, 1:, :], dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, input.size(-1))
                )], dim=1)

        if self.lstm is not None:
            lengths = word_attention_mask.sum(1) + 1  # +cls
            if torch_version >= pack_padded_sequence_cpu_version:
                lengths = lengths.cpu()
            input = pack_padded_sequence(input, lengths, True, False)
            input, _ = self.lstm(input)
            input, _ = pad_packed_sequence(input, True, total_length=word_attention_mask.shape[1] + 1)
            input = self.lstm_dropout(input)

        bin_d = self.mlp_bin_d(input)
        bin_h = self.mlp_bin_h(input)
        bin_g = self.mlp_bin_g(input)

        arc_h = self.mlp_arc_h(input)
        arc_d = self.mlp_arc_d(input)

        rel_h = self.mlp_rel_h(input)
        rel_d = self.mlp_rel_d(input)

        # [batch_size, seq_len, seq_len, n_labels]
        s_sib = self.sib_attn(bin_d, bin_d, bin_h).triu_()
        s_sib = (s_sib + s_sib.transpose(-1, -2)).permute(0, 3, 1, 2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_cop = self.cop_attn(bin_h, bin_d, bin_h).permute(0, 3, 1, 2).triu_()
        s_cop = s_cop + s_cop.transpose(-1, -2)
        # [batch_size, seq_len, seq_len, n_labels]
        s_grd = self.grd_attn(bin_g, bin_d, bin_h).permute(0, 3, 1, 2)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_atten(arc_d, arc_h).squeeze_(1)
        # [batch_size, seq_len, seq_len, n_labels]
        s_rel = self.rel_atten(rel_d, rel_h).permute(0, 2, 3, 1)

        loss = None

        # cat cls
        mask = torch.cat([word_attention_mask[:, :1], word_attention_mask], dim=1)
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0

        if labels is not None:
            rel_loss = nn.CrossEntropyLoss()

            head = torch.cat([torch.zeros_like(head[:, :1, :], device=head.device), head], dim=1)
            arc_mask = head.gt(0) & mask
            arc_loss, arc_logits = self.vi((s_arc, s_sib, s_cop, s_grd), mask, head)

            labels = torch.cat([torch.zeros_like(labels[:, :1, :], device=labels.device), labels], dim=1)
            rel_loss = rel_loss(s_rel[arc_mask], labels[arc_mask])
            loss = self.loss_interpolation * rel_loss + (1 - self.loss_interpolation) * arc_loss
        else:
            arc_logits = self.vi((s_arc, s_sib, s_cop, s_grd), mask)

        return GraphResult(loss=loss, arc_logits=arc_logits, rel_logits=s_rel, src_arc_logits=s_arc)


class TransformerVi(BaseModule):
    def __init__(self, hparams, config=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.transformer = AutoModel.from_pretrained(self.hparams.transformer)
        else:
            self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(self.hparams.dropout)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = ViClassifier(
            input_size=hidden_size,
            label_num=self.hparams.num_labels,
            dropout=self.hparams.dropout,
            lstm_num_layers=self.hparams.lstm_num_layers,
            lstm_hidden_size=self.hparams.lstm_hidden_size,
            bin_hidden_size=self.hparams.bin_hidden_size,
            arc_hidden_size=self.hparams.arc_hidden_size,
            rel_hidden_size=self.hparams.rel_hidden_size,
            loss_interpolation=self.hparams.loss_interpolation,
            inference=self.hparams.inference,
            max_iter=self.hparams.max_iter,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--lstm_num_layers', type=int, default=0)
        parser.add_argument('--lstm_hidden_size', type=int, default=600)
        parser.add_argument('--arc_hidden_size', type=int, default=600)
        parser.add_argument('--rel_hidden_size', type=int, default=600)
        parser.add_argument('--bin_hidden_size', type=int, default=150)
        parser.add_argument('--loss_interpolation', type=float, default=0.1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_labels', type=int)
        parser.add_argument('--inference', type=str, default='mfvi')
        parser.add_argument('--max_iter', type=int, default=3)
        return parser

    def forward(
            self,
            input_ids=None,
            logits_mask=None,
            attention_mask=None,
            word_index=None,
            word_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            head=None,
            labels=None
    ) -> GraphResult:
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
            word_index=word_index,
            word_attention_mask=word_attention_mask,
            head=head,
            labels=labels
        )
