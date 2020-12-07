from argparse import ArgumentParser

from torch import nn
from transformers import AutoModel

from collections import namedtuple
from ltp.nn import BaseModule

TokenClassifierResult = namedtuple('TokenClassifierResult', ['loss', 'logits'])


class LinearClassifier(nn.Linear):
    def __init__(self, hidden_size, num_labels):
        super().__init__(hidden_size, num_labels)

    def forward(self, input, attention_mask=None, logits_mask=None, labels=None):
        logits = super(LinearClassifier, self).forward(input)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if logits_mask is None:
                logits_mask = attention_mask[:, 2:] == 1
            if logits_mask is not None:
                active_loss = logits_mask.view(-1)
                active_logits = logits.view(-1, self.out_features)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.out_features), labels.view(-1))

        return TokenClassifierResult(loss=loss, logits=logits)


class TransformerLinear(BaseModule):
    def __init__(self, hparams, config=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.transformer = AutoModel.from_pretrained(self.hparams.transformer)
        else:
            self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(self.hparams.dropout)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = LinearClassifier(hidden_size, self.hparams.num_labels)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_labels', type=int)
        return parser

    def forward(
            self,
            input_ids=None,
            logits_mask=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None
    ):
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
            attention_mask=attention_mask,
            logits_mask=logits_mask,
            labels=labels
        )
