#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from argparse import ArgumentParser

from torch import nn
from transformers import AutoModel

from ltp.nn import BaseModule
from ltp.transformer_linear import LinearClassifier, MLPClassfier
from ltp.transformer_rel_linear import RelativeTransformerLinearClassifier
from ltp.transformer_biaffine import BiaffineClassifier, dep_loss, sdp_loss
from ltp.transformer_vi import ViClassifier
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

        transformer_hidden_size = self.transformer.config.hidden_size
        max_length = self.transformer.config.max_position_embeddings

        if self.hparams.seg_num_labels:
            if self.hparams.seg_use_mlp:
                self.seg_classifier = MLPClassfier(
                    input_size=transformer_hidden_size,
                    hidden_sizes=self.hparams.seg_hidden_sizes,
                    num_labels=self.hparams.seg_num_labels,
                    dropout=self.hparams.dropout,
                    use_crf=self.hparams.seg_use_crf,
                    use_cls=self.hparams.seg_use_cls,
                    use_sep=self.hparams.seg_use_sep,
                    crf_reduction=self.hparams.seg_crf_reduction
                )
            else:
                self.seg_classifier = LinearClassifier(
                    input_size=transformer_hidden_size,
                    num_labels=self.hparams.seg_num_labels
                )
        if self.hparams.pos_num_labels:
            if self.hparams.pos_use_mlp:
                self.pos_classifier = MLPClassfier(
                    input_size=transformer_hidden_size,
                    hidden_sizes=self.hparams.pos_hidden_sizes,
                    num_labels=self.hparams.pos_num_labels,
                    dropout=self.hparams.dropout,
                    use_crf=self.hparams.pos_use_crf,
                    use_cls=self.hparams.pos_use_cls,
                    use_sep=self.hparams.pos_use_sep,
                    crf_reduction=self.hparams.pos_crf_reduction
                )
            else:
                self.pos_classifier = LinearClassifier(
                    input_size=transformer_hidden_size,
                    num_labels=self.hparams.pos_num_labels
                )
        if self.hparams.ner_num_labels:
            self.ner_classifier = RelativeTransformerLinearClassifier(
                input_size=transformer_hidden_size,
                hidden_size=self.hparams.ner_hidden_size,
                num_layers=self.hparams.ner_num_layers,
                num_heads=self.hparams.ner_num_heads,
                num_labels=self.hparams.ner_num_labels,
                dropout=self.hparams.dropout,
                max_length=max_length,
                use_crf=self.hparams.ner_use_crf,
                use_cls=self.hparams.ner_use_cls,
                use_sep=self.hparams.ner_use_sep,
                crf_reduction=self.hparams.ner_crf_reduction,
                disable_relative_transformer=self.hparams.ner_disable_relative_transformer
            )
        if self.hparams.dep_num_labels:
            self.dep_classifier = BiaffineClassifier(
                input_size=transformer_hidden_size,
                label_num=self.hparams.dep_num_labels,
                arc_hidden_size=self.hparams.dep_arc_hidden_size,
                rel_hidden_size=self.hparams.dep_rel_hidden_size,
                loss_interpolation=self.hparams.dep_loss_interpolation,
                dropout=self.hparams.dropout,
                loss_func=dep_loss,
                char_based=self.hparams.dep_char_based
            )
        if self.hparams.sdp_num_labels:
            if self.hparams.sdp_use_vi:
                self.sdp_classifier = ViClassifier(
                    input_size=transformer_hidden_size,
                    label_num=self.hparams.sdp_num_labels,
                    dropout=self.hparams.dropout,
                    lstm_num_layers=self.hparams.sdp_lstm_num_layers,
                    lstm_hidden_size=self.hparams.sdp_lstm_hidden_size,
                    bin_hidden_size=self.hparams.sdp_bin_hidden_size,
                    arc_hidden_size=self.hparams.sdp_arc_hidden_size,
                    rel_hidden_size=self.hparams.sdp_rel_hidden_size,
                    loss_interpolation=self.hparams.sdp_loss_interpolation,
                    inference=self.hparams.sdp_inference,
                    max_iter=self.hparams.sdp_max_iter,
                )
            else:
                self.sdp_classifier = BiaffineClassifier(
                    input_size=transformer_hidden_size,
                    label_num=self.hparams.sdp_num_labels,
                    arc_hidden_size=self.hparams.sdp_arc_hidden_size,
                    rel_hidden_size=self.hparams.sdp_rel_hidden_size,
                    loss_interpolation=self.hparams.sdp_loss_interpolation,
                    dropout=self.hparams.dropout,
                    loss_func=sdp_loss,
                    char_based=self.hparams.sdp_char_based
                )
        if self.hparams.srl_num_labels:
            self.srl_classifier = BiaffineCRFClassifier(
                input_size=transformer_hidden_size,
                label_num=self.hparams.srl_num_labels,
                hidden_size=self.hparams.srl_hidden_size,
                dropout=self.hparams.dropout
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
        parser.add_argument('--transformer', type=str, default="hfl/chinese-electra-base-discriminator")
        parser.add_argument('--seg_num_labels', type=int, default=2)
        parser.add_argument('--seg_hidden_sizes', nargs='+', type=int)
        parser.add_argument('--seg_use_mlp', action='store_true')
        parser.add_argument('--seg_use_crf', action='store_true')
        parser.add_argument('--seg_use_cls', action='store_true')
        parser.add_argument('--seg_use_sep', action='store_true')
        parser.add_argument('--seg_crf_reduction', type=str, default='sum')

        parser.add_argument('--pos_num_labels', type=int, default=27)
        parser.add_argument('--pos_hidden_sizes', nargs='+', type=int)
        parser.add_argument('--pos_use_mlp', action='store_true')
        parser.add_argument('--pos_use_crf', action='store_true')
        parser.add_argument('--pos_use_cls', action='store_true')
        parser.add_argument('--pos_use_sep', action='store_true')
        parser.add_argument('--pos_crf_reduction', type=str, default='sum')

        parser.add_argument('--ner_num_labels', type=int, default=13)
        parser.add_argument('--ner_use_crf', action='store_true')
        parser.add_argument('--ner_use_cls', action='store_true')
        parser.add_argument('--ner_use_sep', action='store_true')
        parser.add_argument('--ner_disable_relative_transformer', action='store_true')
        parser.add_argument('--ner_crf_reduction', type=str, default='sum')
        parser.add_argument('--ner_num_heads', type=int, default=4)
        parser.add_argument('--ner_num_layers', type=int, default=2)
        parser.add_argument('--ner_hidden_size', type=int, default=256)

        parser.add_argument('--dep_char_based', action='store_true')
        parser.add_argument('--dep_num_labels', type=int, default=14)
        parser.add_argument('--dep_arc_hidden_size', type=int, default=500)
        parser.add_argument('--dep_rel_hidden_size', type=int, default=100)
        parser.add_argument('--dep_loss_interpolation', type=float, default=0.4)

        parser.add_argument('--sdp_use_vi', action='store_true')
        parser.add_argument('--sdp_char_based', action='store_true')
        parser.add_argument('--sdp_inference', type=str, default='mfvi')
        parser.add_argument('--sdp_max_iter', type=int, default=3)
        parser.add_argument('--sdp_num_labels', type=int, default=56)
        parser.add_argument('--sdp_lstm_num_layers', type=int, default=0)
        parser.add_argument('--sdp_lstm_hidden_size', type=int, default=600)
        parser.add_argument('--sdp_arc_hidden_size', type=int, default=600)
        parser.add_argument('--sdp_bin_hidden_size', type=int, default=150)
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
            attention_mask=None,
            word_index=None,
            word_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            **kwargs
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
        sequence_output = self.dropout(sequence_output)

        if task == 'seg':
            return self.seg_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                **kwargs
            )
        elif task == 'pos':
            return self.pos_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                **kwargs
            )
        elif task == 'ner':
            return self.ner_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                **kwargs
            )
        elif task == 'dep':
            return self.dep_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                **kwargs
            )
        elif task == 'sdp':
            return self.sdp_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                **kwargs
            )
        elif task == 'srl':
            return self.srl_classifier(
                input=sequence_output,
                attention_mask=attention_mask,
                word_index=word_index,
                word_attention_mask=word_attention_mask,
                **kwargs
            )
