from argparse import ArgumentParser

import torch
from torch import nn
from transformers import AutoModel

from ltp.nn import BaseModule, RelativeTransformer
from ltp.transformer_linear import TokenClassifierResult


class RelativeTransformerLinearClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_labels, max_length, dropout):
        super().__init__()

        self.relative_transformer = RelativeTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_length * 2
        )
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(self, input, word_index=None, word_attention_mask=None, labels=None,
                return_dict=None, hidden_states=None):
        if word_index is not None:
            input = torch.gather(
                input, dim=1, index=word_index.unsqueeze(-1).expand(-1, -1, input.size(-1))
            )

        sequence_output = self.relative_transformer(input, word_attention_mask)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if word_attention_mask is not None:
                active_loss = word_attention_mask.view(-1)
                active_logits = logits.view(-1, self.classifier.out_features)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return TokenClassifierResult(loss=loss, logits=logits)


class TransformerRelLinear(BaseModule):
    def __init__(self, hparams, config=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.transformer = AutoModel.from_pretrained(self.hparams.transformer)
        else:
            self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(self.hparams.dropout)
        hidden_size = self.transformer.config.hidden_size
        max_length = self.transformer.config.max_position_embeddings
        self.classifier = RelativeTransformerLinearClassifier(
            input_size=hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            max_length=max_length,
            num_labels=self.hparams.num_labels
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_labels', type=int)
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
            sequence_output,
            word_index=word_index,
            word_attention_mask=word_attention_mask,
            labels=labels
        )
