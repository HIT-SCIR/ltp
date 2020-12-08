from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from collections import namedtuple
from ltp.nn import BaseModule, MLP, Bilinear, CRF

SRLResult = namedtuple('SRLResult', ['loss', 'rel_logits', 'decoded', 'labels', 'transitions'])


class BiaffineCRFClassifier(nn.Module):
    def __init__(self, input_size, label_num, dropout, hidden_size=300, eval_transitions=False):
        super().__init__()
        self.label_num = label_num
        self.mlp_rel_h = MLP(input_size, hidden_size, dropout, activation=nn.ReLU)
        self.mlp_rel_d = MLP(input_size, hidden_size, dropout, activation=nn.ReLU)

        self.rel_atten = Bilinear(hidden_size, hidden_size, label_num, bias_x=True, bias_y=True, expand=True)
        self.rel_crf = CRF(label_num)
        self.eval_transitions = eval_transitions

    def rel_forword(self, input):
        rel_h = self.mlp_rel_h(input)
        rel_d = self.mlp_rel_d(input)

        s_rel = self.rel_atten(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_rel

    def forward(self, input, logits_mask=None, word_index=None, word_attention_mask=None, labels=None):
        if word_index is not None:
            input = torch.gather(input, dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, input.size(-1)))

        s_rel = self.rel_forword(input)

        loss = None
        if logits_mask is None:
            logits_mask = word_attention_mask

        mask = logits_mask.unsqueeze(-1).expand(-1, -1, logits_mask.size(1))
        mask = mask & torch.transpose(mask, -1, -2)
        mask = mask.flatten(end_dim=1)
        index = mask[:, 0]

        mask = mask[index]
        s_rel = s_rel.flatten(end_dim=1)[index]
        crf_rel = F.log_softmax(s_rel, dim=-1)

        if labels is not None:
            labels = labels.flatten(end_dim=1)[index]

        loss, decoded = None, None
        if labels is not None:
            loss = - self.rel_crf.forward(emissions=crf_rel, tags=labels, mask=mask)

        if not self.training:
            decoded = self.rel_crf.decode(emissions=crf_rel, mask=mask)

        return SRLResult(
            loss=loss, rel_logits=s_rel, decoded=decoded, labels=labels,
            transitions=(
                self.rel_crf.start_transitions,
                self.rel_crf.transitions,
                self.rel_crf.end_transitions
            )
        )


class TransformerBiaffineCRF(BaseModule):
    def __init__(self, hparams, config=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.transformer = AutoModel.from_pretrained(self.hparams.transformer)
        else:
            self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(self.hparams.dropout)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = BiaffineCRFClassifier(
            hidden_size,
            label_num=self.hparams.num_labels,
            dropout=self.hparams.dropout,
            hidden_size=self.hparams.hidden_size,
            eval_transitions=self.hparams.eval_transitions != 0
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--hidden_size', type=int, default=300)
        parser.add_argument('--loss_interpolation', type=float, default=0.4)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--eval_transitions', type=int, default=0)
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
            labels=None
    ) -> SRLResult:
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
        sequence_output = sequence_output[:, 1:-1, :]
        sequence_output = self.dropout(sequence_output)

        return self.classifier(
            input=sequence_output,
            logits_mask=logits_mask,
            word_index=word_index,
            word_attention_mask=word_attention_mask,
            labels=labels
        )
