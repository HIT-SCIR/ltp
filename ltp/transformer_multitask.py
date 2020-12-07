from argparse import ArgumentParser

from torch import nn
from transformers import AutoModel
from ltp.nn import BaseModule
from ltp.transformer_linear import LinearClassifier
from ltp.transformer_rel_linear import RelativeTransformerLinearClassifier
from ltp.transformer_biaffine import BiaffineClassifier, dep_loss, sdp_loss
from ltp.transformer_biaffine_crf import BiaffineCRFClassifier


class TransformerMultiTask(BaseModule):
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

        if self.hparams.seg_num_labels:
            self.seg_classifier = LinearClassifier(hidden_size, self.hparams.seg_num_labels)
        if self.hparams.pos_num_labels:
            self.pos_classifier = LinearClassifier(hidden_size, self.hparams.pos_num_labels)
        if self.hparams.ner_num_labels:
            self.ner_classifier = RelativeTransformerLinearClassifier(
                input_size=hidden_size,
                hidden_size=self.hparams.ner_hidden_size,
                num_layers=self.hparams.ner_num_layers,
                num_heads=self.hparams.ner_num_heads,
                num_labels=self.hparams.ner_num_labels,
                dropout=self.hparams.dropout,
                max_length=max_length
            )
        if self.hparams.dep_num_labels:
            self.dep_classifier = BiaffineClassifier(
                input_size=hidden_size,
                label_num=self.hparams.dep_num_labels,
                arc_hidden_size=self.hparams.dep_arc_hidden_size,
                rel_hidden_size=self.hparams.dep_rel_hidden_size,
                loss_interpolation=self.hparams.dep_loss_interpolation,
                dropout=self.hparams.dropout,
                loss_func=dep_loss
            )
        if self.hparams.sdp_num_labels:
            self.sdp_classifier = BiaffineClassifier(
                input_size=hidden_size,
                label_num=self.hparams.sdp_num_labels,
                arc_hidden_size=self.hparams.sdp_arc_hidden_size,
                rel_hidden_size=self.hparams.sdp_rel_hidden_size,
                loss_interpolation=self.hparams.sdp_loss_interpolation,
                dropout=self.hparams.dropout,
                loss_func=sdp_loss
            )
        if self.hparams.srl_num_labels:
            self.srl_classifier = BiaffineCRFClassifier(
                input_size=hidden_size,
                label_num=self.hparams.srl_num_labels,
                hidden_size=self.hparams.srl_hidden_size,
                dropout=self.hparams.dropout
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--seg_num_labels', type=int, default=2)

        parser.add_argument('--pos_num_labels', type=int, default=27)

        parser.add_argument('--ner_num_labels', type=int, default=13)
        parser.add_argument('--ner_num_layers', type=int, default=2)
        parser.add_argument('--ner_hidden_size', type=int, default=256)

        parser.add_argument('--ner_num_heads', type=int, default=4)

        parser.add_argument('--dep_num_labels', type=int, default=14)
        parser.add_argument('--dep_arc_hidden_size', type=int, default=500)
        parser.add_argument('--dep_rel_hidden_size', type=int, default=100)
        parser.add_argument('--dep_loss_interpolation', type=float, default=0.4)

        parser.add_argument('--sdp_num_labels', type=int, default=56)
        parser.add_argument('--sdp_arc_hidden_size', type=int, default=600)
        parser.add_argument('--sdp_rel_hidden_size', type=int, default=600)
        parser.add_argument('--sdp_loss_interpolation', type=float, default=0.4)

        parser.add_argument('--srl_num_labels', type=int, default=97)
        parser.add_argument('--srl_hidden_size', type=int, default=600)

        parser.add_argument('--dropout', type=float, default=0.1)
        return parser

    def forward(
            self,
            task,
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

        if task == 'seg':
            sequence_output = sequence_output[:, 1:-1, :]
            sequence_output = self.dropout(sequence_output)
            return self.seg_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                logits_mask=logits_mask,
                labels=labels
            )
        elif task == 'pos':
            sequence_output = sequence_output[:, 1:-1, :]
            sequence_output = self.dropout(sequence_output)
            return self.pos_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                logits_mask=logits_mask,
                labels=labels
            )
        elif task == 'ner':
            sequence_output = sequence_output[:, 1:-1, :]
            sequence_output = self.dropout(sequence_output)
            return self.ner_classifier(
                sequence_output,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                labels=labels
            )
        elif task == 'dep':
            sequence_output = sequence_output[:, :-1, :]
            sequence_output = self.dropout(sequence_output)
            return self.dep_classifier(
                sequence_output,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                head=head
            )
        elif task == 'sdp':
            sequence_output = sequence_output[:, :-1, :]
            sequence_output = self.dropout(sequence_output)
            return self.sdp_classifier(
                sequence_output,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                head=head,
                labels=labels
            )
        elif task == 'srl':
            sequence_output = sequence_output[:, 1:-1, :]
            sequence_output = self.dropout(sequence_output)
            return self.srl_classifier(
                sequence_output,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                labels=labels
            )
