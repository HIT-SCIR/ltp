#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from packaging.version import parse as version_parse

import torch
from transformers import AutoConfig
from ltp.transformer_multitask import TransformerMultiTask as Model

from ltp import __version__


def load_labels(*paths):
    try:
        labels_path = os.path.join(*paths)
        with open(labels_path, encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        return []


def deploy_model(args, version=__version__):
    parsed_version = version_parse(version)
    assert parsed_version.major == 4

    if parsed_version.minor == 0:
        deploy_model_4_0(args, version)
    elif parsed_version.minor == 1:
        deploy_model_4_1(args, version)


def deploy_model_4_1(args, version):
    from argparse import Namespace
    from pytorch_lightning.core.saving import ModelIO

    fake_parser = ArgumentParser()
    fake_parser = Model.add_model_specific_args(fake_parser)
    hparams = fake_parser.parse_args(args=[])

    try:
        ckpt = torch.load(args.resume_from_checkpoint, map_location='cpu')
    except AttributeError as e:
        if "_gpus_arg_default" in e.args[0]:
            from ltp.patchs import pl_1_2_patch_1_1
            patched_model_path = pl_1_2_patch_1_1(args.resume_from_checkpoint)
            ckpt = torch.load(patched_model_path, map_location='cpu')
        else:
            raise e
    hparams.__dict__.update(ckpt[ModelIO.CHECKPOINT_HYPER_PARAMS_KEY])
    transformer_config = AutoConfig.from_pretrained(hparams.transformer)
    model = Model(hparams, config=transformer_config)
    model.load_state_dict(ckpt['state_dict'])

    model_config = Namespace(**model.hparams)
    # LOAD VOCAB
    pos_labels = load_labels(args.pos_data_dir, 'vocabs', 'xpos.txt')
    ner_labels = load_labels(args.ner_data_dir, 'vocabs', 'bio.txt')
    srl_labels = load_labels(args.srl_data_dir, 'vocabs', 'arguments.txt')
    dep_labels = load_labels(args.dep_data_dir, 'vocabs', 'deprel.txt')
    sdp_labels = load_labels(args.sdp_data_dir, 'vocabs', 'deps.txt')

    # MODEL CLIP
    if not len(pos_labels):
        del model.pos_classifier
        model_config.pos_num_labels = 0

    if not len(ner_labels):
        del model.ner_classifier
        model_config.ner_num_labels = 0

    if not len(srl_labels):
        del model.srl_classifier
        model_config.srl_num_labels = 0

    if not len(dep_labels):
        del model.dep_classifier
        model_config.dep_num_labels = 0

    if not len(sdp_labels):
        del model.sdp_classifier
        model_config.sdp_num_labels = 0

    model_state_dict = OrderedDict(model.state_dict().items())

    ltp_model = {
        'version': version,
        'model': model_state_dict,
        'model_config': model_config,
        'transformer_config': model.transformer.config.to_dict(),
        'seg': ['I-W', 'B-W'],
        'pos': pos_labels,
        'ner': ner_labels,
        'srl': srl_labels,
        'dep': dep_labels,
        'sdp': sdp_labels,
    }
    os.makedirs(args.ltp_model, exist_ok=True)
    torch.save(ltp_model, os.path.join(args.ltp_model, 'ltp.model'))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    tokenizer.save_pretrained(args.ltp_model)


def deploy_model_4_0(args, version):
    ltp_adapter_mapper = sorted([
        ('transformer', 'pretrained'),
        ('seg_classifier', 'seg_decoder'),
        ('pos_classifier', 'pos_decoder'),
        ('ner_classifier', 'ner_decoder'),
        ('ner_classifier.classifier', 'ner_decoder.mlp'),
        ('ner_classifier.relative_transformer', 'ner_decoder.transformer'),
        ('srl_classifier', 'srl_decoder'),
        ('srl_classifier.rel_atten', 'srl_decoder.biaffine'),
        ('srl_classifier.crf', 'srl_decoder.crf'),
        ('dep_classifier', 'dep_decoder'),
        ('sdp_classifier', 'sdp_decoder'),
    ], key=lambda x: len(x[0]), reverse=True)

    model = Model.load_from_checkpoint(
        args.resume_from_checkpoint, hparams=args
    )
    model_state_dict = OrderedDict(model.state_dict().items())
    for preffix, target_preffix in ltp_adapter_mapper:
        model_state_dict = {
            key.replace(preffix, target_preffix, 1): value
            for key, value in model_state_dict.items()
        }

    pos_labels = load_labels(args.pos_data_dir, 'vocabs', 'xpos.txt')
    ner_labels = load_labels(args.ner_data_dir, 'ner_labels.txt')
    srl_labels = load_labels(args.srl_data_dir, 'srl_labels.txt')
    dep_labels = load_labels(args.dep_data_dir, 'vocabs', 'deprel.txt')
    sdp_labels = load_labels(args.sdp_data_dir, 'vocabs', 'deps.txt')

    ltp_model = {
        'version': '4.0.0',
        'code_version': version,
        'seg': ['I-W', 'B-W'],
        'pos': pos_labels,
        'ner': ner_labels,
        'srl': srl_labels,
        'dep': dep_labels,
        'sdp': sdp_labels,
        'pretrained_config': model.transformer.config,
        'model_config': {
            'class': 'SimpleMultiTaskModel',
            'init': {
                'seg': {'label_num': args.seg_num_labels},
                'pos': {'label_num': args.pos_num_labels},
                'ner': {
                    'label_num': args.ner_num_labels,
                    'decoder': 'RelativeTransformer',
                    'RelativeTransformer': {
                        'num_heads': args.ner_num_heads,
                        'num_layers': args.ner_num_layers,
                        'hidden_size': args.ner_hidden_size,
                        'dropout': args.dropout
                    }
                },
                'dep': {
                    'label_num': args.dep_num_labels, 'decoder': 'Graph',
                    'Graph': {
                        'arc_hidden_size': args.dep_arc_hidden_size,
                        'rel_hidden_size': args.dep_rel_hidden_size,
                        'dropout': args.dropout
                    }
                },
                'sdp': {
                    'label_num': args.sdp_num_labels, 'decoder': 'Graph',
                    'Graph': {
                        'arc_hidden_size': args.sdp_arc_hidden_size,
                        'rel_hidden_size': args.sdp_rel_hidden_size,
                        'dropout': args.dropout
                    }
                },
                'srl': {
                    'label_num': args.srl_num_labels, 'decoder': 'BiLinearCRF',
                    'BiLinearCRF': {
                        'hidden_size': args.srl_hidden_size,
                        'dropout': args.dropout
                    }
                }
            }
        },
        'model': model_state_dict
    }
    os.makedirs(args.ltp_model, exist_ok=True)
    torch.save(ltp_model, os.path.join(args.ltp_model, 'ltp.model'))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    tokenizer.save_pretrained(args.ltp_model)
