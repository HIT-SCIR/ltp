import os
import torch
from packaging.version import parse as version_parse
from ltp.transformer_multitask import TransformerMultiTask as Model

from ltp import __version__


def load_labels(labels_path):
    try:
        with open(labels_path, encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        return []


def deploy_model(args, version=__version__):
    version = version_parse(version)
    assert version.major == 4

    if version.minor == 0:
        deploy_model_4_0(args)
    elif version.minor == 1:
        deploy_model_4_1(args)


def deploy_model_4_1(args):
    from argparse import Namespace

    model = Model.load_from_checkpoint(
        args.resume_from_checkpoint, hparams=args
    )
    model_state_dict = model.state_dict()
    model_config = Namespace(**model.hparams)

    ltp_model = {
        'version': "4.1.0",
        'model': model_state_dict,
        'model_config': model_config,
        'transformer_config': model.transformer.config.to_dict(),
        'seg': ['I-W', 'B-W'],
        'pos': load_labels(os.path.join(args.pos_data_dir, 'pos_labels.txt')),
        'ner': load_labels(os.path.join(args.ner_data_dir, 'ner_labels.txt')),
        'srl': load_labels(os.path.join(args.srl_data_dir, 'srl_labels.txt')),
        'dep': load_labels(os.path.join(args.dep_data_dir, 'dep_labels.txt')),
        'sdp': load_labels(os.path.join(args.sdp_data_dir, 'deps_labels.txt')),
    }
    os.makedirs(args.ltp_model, exist_ok=True)
    torch.save(ltp_model, os.path.join(args.ltp_model, 'ltp.model'))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    tokenizer.save_pretrained(args.ltp_model)


def deploy_model_4_0(args):
    ltp_adapter_mapper = sorted([
        ('transformer', 'pretrained'),
        ('seg_classifier', 'seg_decoder'),
        ('pos_classifier', 'pos_decoder'),
        ('ner_classifier', 'ner_decoder'),
        ('ner_classifier.classifier', 'ner_decoder.mlp'),
        ('ner_classifier.relative_transformer', 'ner_decoder.transformer'),
        ('srl_classifier', 'srl_decoder'),
        ('srl_classifier.rel_atten', 'srl_decoder.biaffine'),
        ('srl_classifier.rel_crf', 'srl_decoder.crf'),
        ('dep_classifier', 'dep_decoder'),
        ('sdp_classifier', 'sdp_decoder'),
    ], key=lambda x: len(x[0]), reverse=True)

    model = Model.load_from_checkpoint(
        args.resume_from_checkpoint, hparams=args
    )
    model_state_dict = model.state_dict()
    for preffix, target_preffix in ltp_adapter_mapper:
        model_state_dict = {
            key.replace(preffix, target_preffix, 1): value
            for key, value in model_state_dict.items()
        }

    ltp_model = {
        'version': '4.0.0',
        'seg': ['I-W', 'B-W'],
        'pos': load_labels(os.path.join(args.pos_data_dir, 'pos_labels.txt')),
        'ner': load_labels(os.path.join(args.ner_data_dir, 'ner_labels.txt')),
        'srl': load_labels(os.path.join(args.srl_data_dir, 'srl_labels.txt')),
        'dep': load_labels(os.path.join(args.dep_data_dir, 'dep_labels.txt')),
        'sdp': load_labels(os.path.join(args.sdp_data_dir, 'deps_labels.txt')),
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
