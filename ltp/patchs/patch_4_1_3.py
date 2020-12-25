#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from argparse import Namespace
from packaging import version

_patch_version = version.parse('4.1.3')


def patch(ckpt):
    if 'version' in ckpt and version.parse(ckpt['version']) < _patch_version:
        srl_decoder_mlp_rel_h = {}
        srl_decoder_mlp_rel_d = {}

        model_config: Namespace = ckpt['model_config']
        setattr(model_config, 'ner_use_crf', False)
        setattr(model_config, 'ner_crf_reduction', 'sum')

        for key, value in ckpt['model'].items():
            key: str
            if key.startswith('srl_classifier.mlp_rel_h'):
                srl_decoder_mlp_rel_h[key] = value
            elif key.startswith('srl_classifier.mlp_rel_d'):
                srl_decoder_mlp_rel_d[key] = value
        srl_decoder_mlp_rel_h = {
            k.replace('srl_classifier.mlp_rel_h', 'srl_classifier.mlp_rel_d'): v for k, v in
            srl_decoder_mlp_rel_h.items()
        }
        srl_decoder_mlp_rel_d = {
            k.replace('srl_classifier.mlp_rel_d', 'srl_classifier.mlp_rel_h'): v for k, v in
            srl_decoder_mlp_rel_d.items()
        }

        try:
            ckpt['model']['srl_classifier.crf.start_transitions'] \
                = ckpt['model'].pop('srl_classifier.rel_crf.start_transitions')
            ckpt['model']['srl_classifier.crf.end_transitions'] \
                = ckpt['model'].pop('srl_classifier.rel_crf.end_transitions')
            ckpt['model']['srl_classifier.crf.transitions'] \
                = ckpt['model'].pop('srl_classifier.rel_crf.transitions')
        except Exception as e:
            pass

        ckpt['model'].update(srl_decoder_mlp_rel_h)
        ckpt['model'].update(srl_decoder_mlp_rel_d)
