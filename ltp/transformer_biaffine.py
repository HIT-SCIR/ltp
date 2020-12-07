from argparse import ArgumentParser

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from collections import namedtuple
from ltp.nn import MLP, Bilinear, BaseModule

GraphResult = namedtuple('GraphResult', ['loss', 'arc_logits', 'rel_logits'])


def dep_loss(model, s_arc, s_rel, head, labels, logits_mask):
    head_loss = nn.CrossEntropyLoss()
    rel_loss = nn.CrossEntropyLoss()

    # ignore the first token of each sentence
    s_arc = s_arc[:, 1:, :]
    s_rel = s_rel[:, 1:, :]

    # Only keep active parts of the loss
    active_heads = head[logits_mask]
    active_labels = labels[logits_mask]
    s_arc, s_rel = s_arc[logits_mask], s_rel[logits_mask]

    s_rel = s_rel[torch.arange(len(active_heads)), active_heads]

    arc_loss = head_loss(s_arc, active_heads)
    rel_loss = rel_loss(s_rel, active_labels)
    loss = 2 * ((1 - model.loss_interpolation) * arc_loss + model.loss_interpolation * rel_loss)

    return loss


def sdp_loss(model, s_arc, s_rel, head, labels, logits_mask):
    head_loss = nn.BCEWithLogitsLoss()
    rel_loss = nn.CrossEntropyLoss()

    # ignore the first token of each sentence
    s_arc = s_arc[:, 1:, :]
    s_rel = s_rel[:, 1:, :]

    # mask
    mask = logits_mask.unsqueeze(-1).expand_as(s_arc)

    arc_loss = head_loss(s_arc[mask], head[mask].float())
    rel_loss = rel_loss(s_rel[head > 0], labels[head > 0])

    loss = 2 * ((1 - model.loss_interpolation) * arc_loss + model.loss_interpolation * rel_loss)

    return loss


class BiaffineClassifier(nn.Module):
    def __init__(self, input_size, label_num, dropout,
                 arc_hidden_size=500, rel_hidden_size=100, loss_interpolation=0.4, loss_func=dep_loss):
        super().__init__()
        self.label_num = label_num
        self.loss_interpolation = loss_interpolation
        self.mlp_arc_h = MLP(input_size, arc_hidden_size, dropout, activation=nn.ReLU)
        self.mlp_arc_d = MLP(input_size, arc_hidden_size, dropout, activation=nn.ReLU)
        self.mlp_rel_h = MLP(input_size, rel_hidden_size, dropout, activation=nn.ReLU)
        self.mlp_rel_d = MLP(input_size, rel_hidden_size, dropout, activation=nn.ReLU)

        self.arc_atten = Bilinear(arc_hidden_size, arc_hidden_size, 1, bias_x=True, bias_y=False, expand=True)
        self.rel_atten = Bilinear(rel_hidden_size, rel_hidden_size, label_num, bias_x=True, bias_y=True, expand=True)

        self.loss_func = loss_func

    def forward(self, input, logits_mask=None, word_index=None, word_attention_mask=None, head=None, labels=None):
        if word_index is not None:
            input = torch.cat([input[:, :1, :], torch.gather(
                input[:, 1:, :], dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, input.size(-1))
            )], dim=1)

        arc_h = self.mlp_arc_h(input)
        arc_d = self.mlp_arc_d(input)

        rel_h = self.mlp_rel_h(input)
        rel_d = self.mlp_rel_d(input)

        s_arc = self.arc_atten(arc_d, arc_h).squeeze_(1)
        s_rel = self.rel_atten(rel_d, rel_h).permute(0, 2, 3, 1)

        loss = None
        if labels is not None:
            if logits_mask is None:
                logits_mask = word_attention_mask
            loss = self.loss_func(self, s_arc, s_rel, head, labels, logits_mask)

        if word_attention_mask is not None:
            activate_word_mask = torch.cat([word_attention_mask[:, :1], word_attention_mask], dim=1)
            activate_word_mask = activate_word_mask.unsqueeze(-1).expand_as(s_arc)
            activate_word_mask = activate_word_mask & activate_word_mask.transpose(-1, -2)
            s_arc.masked_fill_(~activate_word_mask, float('-inf'))

        return GraphResult(loss=loss, arc_logits=s_arc, rel_logits=s_rel)


class TransformerBiaffine(BaseModule):
    def __init__(self, hparams, loss_func=dep_loss, config=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.transformer = AutoModel.from_pretrained(self.hparams.transformer)
        else:
            self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(self.hparams.dropout)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = BiaffineClassifier(
            hidden_size,
            label_num=self.hparams.num_labels,
            dropout=self.hparams.dropout,
            arc_hidden_size=self.hparams.arc_hidden_size,
            rel_hidden_size=self.hparams.rel_hidden_size,
            loss_interpolation=self.hparams.loss_interpolation,
            loss_func=loss_func
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--arc_hidden_size', type=int, default=500)
        parser.add_argument('--rel_hidden_size', type=int, default=200)
        parser.add_argument('--loss_interpolation', type=float, default=0.4)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_labels', type=int)
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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
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
        sequence_output = sequence_output[:, :-1, :]
        sequence_output = self.dropout(sequence_output)

        return self.classifier(
            input=sequence_output,
            logits_mask=logits_mask,
            word_index=word_index,
            word_attention_mask=word_attention_mask,
            head=head,
            labels=labels
        )
