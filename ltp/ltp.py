#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import os
import torch
import itertools
import regex as re
from typing import List

from transformers import AutoTokenizer, cached_path
from transformers.file_utils import is_remote_url

from ltp.models import Model
from ltp.utils import length_to_mask, eisner, is_chinese_char, split_sentence
from ltp.utils.seqeval import get_entities

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )

default_cache_path = os.path.join(torch_cache_home, "ltp")
LTP_CACHE = os.getenv("LTP_CACHE", default_cache_path)

model_map = {
    'base': 'http://39.96.43.154/ltp/v2/base.tgz',
    'small': 'http://39.96.43.154/ltp/v2/small.tgz',
    'tiny': 'http://39.96.43.154/ltp/v2/tiny.tgz'
}


def no_gard(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


WORD_START = 'B-W'
WORD_MIDDLE = 'I-W'


class LTP(object):
    model: Model
    seg_vocab: List[str]
    pos_vocab: List[str]
    ner_vocab: List[str]
    dep_vocab: List[str]
    sdp_vocab: List[str]
    srl_vocab: List[str]

    def __init__(self, path: str = 'small', batch_size: int = 10, device=None, vocab: str = None, **kwargs):
        if device is not None:
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if path in model_map or is_remote_url(path) or os.path.isfile(path):
            proxies = kwargs.pop("proxies", None)
            cache_dir = kwargs.pop("cache_dir", LTP_CACHE)
            force_download = kwargs.pop("force_download", False)
            resume_download = kwargs.pop("resume_download", False)
            local_files_only = kwargs.pop("local_files_only", False)
            path = cached_path(
                model_map.get(path, path),
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                extract_compressed_file=True
            )
        elif not os.path.isdir(path):
            raise FileNotFoundError()
        ckpt = torch.load(os.path.join(path, "ltp.model"), map_location=self.device)
        ckpt['model_config']['init'].pop('pretrained')
        self.model = Model.from_params(ckpt['model_config'], config=ckpt['pretrained_config']).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.seg_vocab = [WORD_START, WORD_MIDDLE]
        self.pos_vocab = ckpt['pos']
        self.ner_vocab = ckpt['ner']
        self.dep_vocab = ckpt['dep']
        self.sdp_vocab = ckpt['sdp']
        self.srl_vocab = [re.sub(r'ARG(\d)', r'A\1', tag) for tag in ckpt['srl']]
        self.split = lambda a: map(lambda b: a[b:b + batch_size], range(0, len(a), batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(path, config=self.model.pretrained.config, use_fast=True)

    def _convert_idx_to_name(self, y, array_len, id2label):
        if id2label:
            return [[id2label[idx] for idx in row[:row_len]] for row, row_len in zip(y, array_len)]
        else:
            return [[idx for idx in row[:row_len]] for row, row_len in zip(y, array_len)]

    def _get_entities_with_list(self, labels_, itos):
        res = []
        for labels in labels_:
            labels = [itos[label] for label in labels]
            labels = get_entities(labels)
            res.append(labels)
        return res

    def _get_graph_entities(self, arcs, labels, itos):
        sequence_num = labels.size(0)
        arcs = torch.nonzero(arcs, as_tuple=False).cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy()

        res = [[] for _ in range(sequence_num)]
        for idx, arc_s, arc_e in arcs:
            label = labels[idx, arc_s, arc_e]
            res[idx].append((arc_s, arc_e, itos[label]))

        return res

    def sent_split(self, inputs: List[str], flag: str = "all", limit: int = 510):
        inputs = [split_sentence(text, flag=flag, limit=limit) for text in inputs]
        inputs = list(itertools.chain(*inputs))
        return inputs

    @no_gard
    def seg(self, inputs: List[str]):
        length = torch.as_tensor([len(text) for text in inputs], device=self.device)
        tokenizerd = self.tokenizer.batch_encode_plus(inputs, return_tensors='pt', padding=True)
        pretrained_output, *_ = self.model.pretrained(
            input_ids=tokenizerd['input_ids'].to(self.device),
            attention_mask=tokenizerd['attention_mask'].to(self.device),
            token_type_ids=tokenizerd['token_type_ids'].to(self.device)
        )

        # remove [CLS] [SEP]
        word_cls = pretrained_output[:, :1]
        char_input = torch.narrow(pretrained_output, 1, 1, pretrained_output.size(1) - 2)

        segment_output = torch.argmax(self.model.seg_decoder(char_input), dim=-1).cpu().numpy()
        segment_output = self._convert_idx_to_name(segment_output, length, self.seg_vocab)

        sentences = []
        word_idx = []
        word_length = []
        for source_text, encoding, sentence_seg_tag in zip(inputs, tokenizerd.encodings, segment_output):
            text = [source_text[start:end] for start, end in encoding.offsets[1:-1] if end != 0]

            last_word = 0
            for idx, word in enumerate(encoding.words[1:-1]):
                if word is None or is_chinese_char(text[idx][-1]):
                    continue
                if word != last_word:
                    text[idx] = ' ' + text[idx]
                    last_word = word
                else:
                    sentence_seg_tag[idx] = WORD_MIDDLE

            entities = get_entities(sentence_seg_tag)
            word_length.append(len(entities))

            sentences.append([''.join(text[entity[1]:entity[2] + 1]).strip() for entity in entities])
            word_idx.append(torch.as_tensor([entity[1] for entity in entities], device=self.device))

        word_idx = torch.nn.utils.rnn.pad_sequence(word_idx, batch_first=True)
        word_idx = word_idx.unsqueeze(-1).expand(-1, -1, char_input.shape[-1])

        word_input = torch.gather(char_input, dim=1, index=word_idx)

        word_cls_input = torch.cat([word_cls, word_input], dim=1)
        word_cls_mask = length_to_mask(torch.as_tensor(word_length, device=self.device) + 1)
        word_cls_mask[:, 0] = False  # ignore the first token of each sentence
        return sentences, {
            'word_cls': word_cls, 'word_input': word_input, 'word_length': word_length,
            'word_cls_input': word_cls_input, 'word_cls_mask': word_cls_mask
        }

    @no_gard
    def pos(self, hidden: dict):
        # 词性标注
        postagger_output = self.model.pos_decoder(hidden['word_input'], hidden['word_length'])
        postagger_output = torch.argmax(postagger_output, dim=-1).cpu().numpy()
        postagger_output = self._convert_idx_to_name(postagger_output, hidden['word_length'], self.pos_vocab)
        return postagger_output

    @no_gard
    def ner(self, hidden: dict):
        # 命名实体识别
        word_length = torch.as_tensor(hidden['word_length'], device=self.device)
        ner_output = self.model.ner_decoder(hidden['word_input'], word_length)
        ner_output = torch.argmax(ner_output, dim=-1).cpu().numpy()
        ner_output = self._convert_idx_to_name(ner_output, hidden['word_length'], self.ner_vocab)
        return [get_entities(ner) for ner in ner_output]

    @no_gard
    def srl(self, hidden: dict, keep_empty=True):
        # 语义角色标注
        word_length = torch.as_tensor(hidden['word_length'], device=hidden['word_input'].device)
        word_mask = length_to_mask(word_length)
        srl_output, srl_length, crf = self.model.srl_decoder(hidden['word_input'], hidden['word_length'])
        mask = word_mask.unsqueeze_(-1).expand(-1, -1, word_mask.size(1))
        mask = (mask & mask.transpose(-1, -2)).flatten(end_dim=1)
        index = mask[:, 0]
        mask = mask[index]

        srl_input = srl_output.flatten(end_dim=1)[index]
        srl_entities = crf.decode(torch.log_softmax(srl_input, dim=-1), mask)
        srl_entities = self._get_entities_with_list(srl_entities, self.srl_vocab)

        srl_labels_res = []
        for length in srl_length:
            srl_labels_res.append([])
            curr_srl_labels, srl_entities = srl_entities[:length], srl_entities[length:]
            srl_labels_res[-1].extend(curr_srl_labels)

        if not keep_empty:
            srl_labels_res = [[(idx, labels) for idx, labels in enumerate(srl_labels) if len(labels)]
                              for srl_labels in srl_labels_res]
        return srl_labels_res

    @no_gard
    def dep(self, hidden: dict, fast=False):
        # 依存句法树
        dep_arc, dep_label, word_length = self.model.dep_decoder(hidden['word_cls_input'], hidden['word_length'])
        if fast:
            dep_arc_fix = dep_arc.argmax(dim=-1).unsqueeze_(-1).expand_as(dep_arc)
        else:
            dep_arc_fix = eisner(dep_arc, hidden['word_cls_mask']).unsqueeze_(-1).expand_as(dep_arc)
        dep_arc = torch.zeros_like(dep_arc, dtype=torch.bool).scatter_(dim=-1, index=dep_arc_fix, value=True)
        dep_label = torch.argmax(dep_label, dim=-1)

        word_cls_mask = hidden['word_cls_mask']
        word_cls_mask = word_cls_mask.unsqueeze(-1).expand(-1, -1, word_cls_mask.size(1))
        dep_arc = dep_arc & word_cls_mask
        dep_label = self._get_graph_entities(dep_arc, dep_label, self.dep_vocab)

        return dep_label

    @no_gard
    def sdp(self, hidden: dict, graph=True):
        # 语义依存
        sdp_arc, sdp_label, _ = self.model.sdp_decoder(hidden['word_cls_input'], hidden['word_length'])
        sdp_arc = torch.sigmoid_(sdp_arc)

        if graph:
            # 语义依存图
            sdp_arc.transpose_(-1, -2)
            sdp_root_mask = sdp_arc[:, 0].argmax(dim=-1).unsqueeze_(-1).expand_as(sdp_arc[:, 0])
            sdp_arc[:, 0] = 0
            sdp_arc[:, 0].scatter_(dim=-1, index=sdp_root_mask, value=1)
            sdp_arc_T = sdp_arc.transpose(-1, -2)
            sdp_arc_fix = sdp_arc_T.argmax(dim=-1).unsqueeze_(-1).expand_as(sdp_arc)
            sdp_arc = ((sdp_arc_T > 0.5) & (sdp_arc_T > sdp_arc)). \
                scatter_(dim=-1, index=sdp_arc_fix, value=True)
        else:
            # 语义依存树
            sdp_arc_fix = eisner(sdp_arc, hidden['word_cls_mask']).unsqueeze_(-1).expand_as(sdp_arc)
            sdp_arc = torch.zeros_like(sdp_arc, dtype=torch.bool).scatter_(dim=-1, index=sdp_arc_fix, value=True)

        sdp_label = torch.argmax(sdp_label, dim=-1)

        word_cls_mask = hidden['word_cls_mask']
        word_cls_mask = word_cls_mask.unsqueeze(-1).expand(-1, -1, word_cls_mask.size(1))
        sdp_arc = sdp_arc & word_cls_mask
        sdp_label = self._get_graph_entities(sdp_arc, sdp_label, self.sdp_vocab)

        return sdp_label
