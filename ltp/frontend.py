#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
import itertools
import regex as re
from packaging import version
from typing import Union, List
from argparse import ArgumentParser

import torch
import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import cached_path, TensorType, BatchEncoding
from transformers.file_utils import is_remote_url

transformers_version = version.parse(transformers.__version__)

from ltp.algorithms import Trie, eisner, split_sentence
from ltp.transformer_multitask import TransformerMultiTask as Model
from ltp.utils import length_to_mask, get_entities, fake_import_pytorch_lightning
from ltp.patchs import model_patch_4_1_3

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
    'base': 'http://39.96.43.154/ltp/v3/base.tgz',
    'base1': 'http://39.96.43.154/ltp/v3/base1.tgz',
    'base2': 'http://39.96.43.154/ltp/v3/base2.tgz',
    'small': 'http://39.96.43.154/ltp/v3/small.tgz',
    'tiny': 'http://39.96.43.154/ltp/v3/tiny.tgz',
    'GSD': 'http://39.96.43.154/ltp/ud/gsd.tgz',
    'GSD+CRF': 'http://39.96.43.154/ltp/ud/gsd_crf.tgz',
    'GSDSimp': 'http://39.96.43.154/ltp/ud/gsd.tgz',
    'GSDSimp+CRF': 'http://39.96.43.154/ltp/ud/gsd_crf.tgz',
}


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


WORD_START = 'B-W'
WORD_MIDDLE = 'I-W'


def get_entities_with_list(labels_, itos):
    res = []
    for labels in labels_:
        labels = [itos[label] for label in labels]
        labels = get_entities(labels)
        res.append(labels)
    return res


def get_graph_entities(arcs, labels, itos):
    sequence_num = labels.size(0)
    arcs = torch.nonzero(arcs, as_tuple=False).cpu().detach().numpy().tolist()
    labels = labels.cpu().detach().numpy()

    res = [[] for _ in range(sequence_num)]
    for idx, arc_s, arc_e in arcs:
        label = labels[idx, arc_s, arc_e]
        res[idx].append((arc_s, arc_e, itos[label]))

    return res


def convert_idx_to_name(y, array_len, id2label):
    if id2label:
        return [[id2label[idx] for idx in row[:row_len]] for row, row_len in zip(y, array_len)]
    else:
        return [[idx for idx in row[:row_len]] for row, row_len in zip(y, array_len)]


class LTP(object):
    model: Model
    seg_vocab: List[str]
    pos_vocab: List[str]
    ner_vocab: List[str]
    dep_vocab: List[str]
    sdp_vocab: List[str]
    srl_vocab: List[str]

    tensor: TensorType = TensorType.PYTORCH

    def __init__(self, path: str = 'small', device=None, **kwargs):
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
        try:
            ckpt = torch.load(os.path.join(path, "ltp.model"), map_location=self.device)
        except Exception as e:
            fake_import_pytorch_lightning()
            ckpt = torch.load(os.path.join(path, "ltp.model"), map_location=self.device)

        model_patch_4_1_3(ckpt)

        self.cache_dir = path
        transformer_config = ckpt['transformer_config']
        transformer_config['torchscript'] = True
        config = AutoConfig.for_model(**transformer_config)

        parser = ArgumentParser()
        parser = Model.add_model_specific_args(parser)
        model_args = parser.parse_args(args=[], namespace=ckpt['model_config'])

        self.model = Model(model_args, config=config).to(self.device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model.eval()

        self.seg_vocab = ckpt.get('seg', [WORD_MIDDLE, WORD_START])
        self.seg_vocab_dict = {tag: idx for idx, tag in enumerate(self.seg_vocab)}
        self.pos_vocab = ckpt.get('pos', [])
        self.ner_vocab = ckpt.get('ner', [])
        self.dep_vocab = ckpt.get('dep', [])
        self.sdp_vocab = ckpt.get('sdp', [])
        self.srl_vocab = [re.sub(r'ARG(\d)', r'A\1', tag.lstrip('ARGM-')) for tag in ckpt.get('srl', [])]
        self.tokenizer = AutoTokenizer.from_pretrained(path, config=self.model.transformer.config, use_fast=True)
        self.trie = Trie()
        self._model_version = ckpt.get('version', None)

    def __str__(self):
        return f"LTP {self.version} on {self.device} (model version: {self.model_version}) "

    def __repr__(self):
        return f"LTP {self.version} on {self.device} (model version: {self.model_version}) "

    @property
    def available_models(self):
        return model_map.keys()

    @property
    def version(self):
        from ltp import __version__ as version
        return version

    @property
    def model_version(self):
        return self._model_version or 'unknown'

    @property
    def max_length(self):
        return self.model.transformer.config.max_position_embeddings

    def init_dict(self, path, max_window=None):
        self.trie.init(path, max_window)

    def add_words(self, words, max_window=None):
        self.trie.add_words(words)
        self.trie.max_window = max_window

    @staticmethod
    def sent_split(inputs: List[str], flag: str = "all", limit: int = 510):
        inputs = [split_sentence(text, flag=flag, limit=limit) for text in inputs]
        inputs = list(itertools.chain(*inputs))
        return inputs

    def seg_with_dict(self, inputs: List[str], tokenized: BatchEncoding, batch_prefix):
        # 进行正向字典匹配
        matching = []
        for source_text, encoding, preffix in zip(inputs, tokenized.encodings, batch_prefix):
            text = [source_text[start:end] for start, end in encoding.offsets[1:-1] if end != 0]
            matching_pos = self.trie.maximum_forward_matching(text, preffix)
            matching.append(matching_pos)
        return matching

    @no_grad
    def _seg(self, tokenizerd, is_preseged=False):
        input_ids = tokenizerd['input_ids'].to(self.device)
        attention_mask = tokenizerd['attention_mask'].to(self.device)
        token_type_ids = tokenizerd['token_type_ids'].to(self.device)
        length = torch.sum(attention_mask, dim=-1) - 2

        pretrained_output, *_ = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        # remove [CLS] [SEP]
        word_cls = pretrained_output[:, :1]
        char_input = torch.narrow(pretrained_output, 1, 1, pretrained_output.size(1) - 2)
        if is_preseged:
            segment_output = None
        else:
            segment_output = self.model.seg_classifier.forward(char_input, is_processed=True)
            segment_output = segment_output.decoded or torch.argmax(segment_output.logits, dim=-1).cpu().numpy()
        return word_cls, char_input, segment_output, length

    @no_grad
    def seg(self, inputs: Union[List[str], List[List[str]]], truncation: bool = True, is_preseged=False):
        """
        分词

        Args:
            inputs: 句子列表
            truncation: 是否对过长的句子进行截断，如果为 False 可能会抛出异常
            is_preseged:  是否已经进行过分词

        Returns:
            words: 分词后的序列
            hidden: 用于其他任务的中间表示
        """

        if transformers_version.major >= 3 and transformers_version.major > 1:
            kwargs = {'is_split_into_words': is_preseged}
        else:
            kwargs = {'is_pretokenized': is_preseged}

        tokenized = self.tokenizer.batch_encode_plus(
            inputs, padding=True, truncation=truncation,
            return_tensors=self.tensor, max_length=self.max_length,
            **kwargs
        )
        cls, hidden, seg, lengths = self._seg(tokenized, is_preseged=is_preseged)

        batch_prefix = [[
            word_idx != encoding.words[idx - 1]
            for idx, word_idx in enumerate(encoding.words) if word_idx is not None
        ] for encoding in tokenized.encodings]

        # merge segments with maximum forward matching
        if self.trie.is_init and not is_preseged:
            matches = self.seg_with_dict(inputs, tokenized, batch_prefix)
            for sent_match, sent_seg in zip(matches, seg):
                for start, end in sent_match:
                    sent_seg[start] = self.seg_vocab_dict[WORD_START]
                    sent_seg[start + 1:end] = self.seg_vocab_dict[WORD_MIDDLE]
                    if end < len(sent_seg):
                        sent_seg[end] = self.seg_vocab_dict[WORD_START]

        if is_preseged:
            sentences = inputs
            word_length = [len(sentence) for sentence in sentences]

            word_idx = []
            for encodings in tokenized.encodings:
                sentence_word_idx = []
                for idx, (start, end) in enumerate(encodings.offsets[1:]):
                    if start == 0 and end:
                        sentence_word_idx.append(idx)
                word_idx.append(torch.as_tensor(sentence_word_idx, device=self.device))
        else:
            segment_output = convert_idx_to_name(seg, lengths, self.seg_vocab)
            sentences = []
            word_idx = []
            word_length = []

            for source_text, length, encoding, seg_tag, preffix in \
                    zip(inputs, lengths, tokenized.encodings, segment_output, batch_prefix):
                offsets = encoding.offsets[1:length + 1]
                text = []
                last_offset = None
                for start, end in offsets:
                    text.append('' if last_offset == (start, end) else source_text[start:end])
                    last_offset = (start, end)

                for idx in range(1, length):
                    current_beg = offsets[idx][0]
                    forward_end = offsets[idx - 1][-1]
                    if forward_end < current_beg:
                        text[idx] = source_text[forward_end:current_beg] + text[idx]
                    if not preffix[idx]:
                        seg_tag[idx] = WORD_MIDDLE

                entities = get_entities(seg_tag)
                word_length.append(len(entities))
                sentences.append([''.join(text[entity[1]:entity[2] + 1]).strip() for entity in entities])
                word_idx.append(torch.as_tensor([entity[1] for entity in entities], device=self.device))

        word_idx = torch.nn.utils.rnn.pad_sequence(word_idx, batch_first=True)
        word_idx = word_idx.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])  # 展开

        word_input = torch.gather(hidden, dim=1, index=word_idx)  # 每个word第一个char的向量

        if len(self.dep_vocab) + len(self.sdp_vocab) > 0:
            word_cls_input = torch.cat([cls, word_input], dim=1)
            word_cls_mask = length_to_mask(torch.as_tensor(word_length, device=self.device) + 1)
            word_cls_mask[:, 0] = False
        else:
            word_cls_input, word_cls_mask = None, None

        return sentences, {
            'word_cls': cls, 'word_input': word_input, 'word_length': word_length,
            'word_cls_input': word_cls_input, 'word_cls_mask': word_cls_mask
        }

    @no_grad
    def pos(self, hidden: dict):
        """
        词性标注
        Args:
            hidden: 分词时所得到的中间表示

        Returns:
            pos: 词性标注结果
        """
        if len(self.pos_vocab) == 0:
            return []
        postagger_output = self.model.pos_classifier(hidden['word_input'], is_processed=True)
        postagger_output = postagger_output.decoded or torch.argmax(postagger_output.logits, dim=-1).cpu().numpy()
        postagger_output = convert_idx_to_name(postagger_output, hidden['word_length'], self.pos_vocab)
        return postagger_output

    @no_grad
    def ner(self, hidden: dict, as_entities=True):
        """
        命名实体识别
        Args:
            hidden: 分词时所得到的中间表示
            as_entities: 是否以 Entity(Type, Start, End) 的形式返回

        Returns:
            pos: 命名实体识别结果
        """
        if len(self.ner_vocab) == 0:
            return []
        ner_output = self.model.ner_classifier.forward(
            hidden['word_input'], word_attention_mask=hidden['word_cls_mask'][:, 1:], is_processed=True
        )
        ner_output = ner_output.decoded or torch.argmax(ner_output.logits, dim=-1).cpu().numpy()
        ner_output = convert_idx_to_name(ner_output, hidden['word_length'], self.ner_vocab)
        return [get_entities(ner) for ner in ner_output] if as_entities else ner_output

    @no_grad
    def srl(self, hidden: dict, keep_empty=True):
        """
        语义角色标注
        Args:
            hidden: 分词时所得到的中间表示

        Returns:
            pos: 语义角色标注结果
        """
        if len(self.srl_vocab) == 0:
            return []
        srl_output = self.model.srl_classifier.forward(
            input=hidden['word_input'],
            word_attention_mask=hidden['word_cls_mask'][:, 1:],
            is_processed=True
        ).decoded
        srl_entities = get_entities_with_list(srl_output, self.srl_vocab)

        srl_labels_res = []
        for length in hidden['word_length']:
            srl_labels_res.append([])
            curr_srl_labels, srl_entities = srl_entities[:length], srl_entities[length:]
            srl_labels_res[-1].extend(curr_srl_labels)

        if not keep_empty:
            srl_labels_res = [
                [(idx, labels) for idx, labels in enumerate(srl_labels) if len(labels)]
                for srl_labels in srl_labels_res
            ]
        return srl_labels_res

    @no_grad
    def dep(self, hidden: dict, fast=False, as_tuple=True):
        """
        依存句法树
        Args:
            hidden: 分词时所得到的中间表示
            fast: 启用 fast 模式时，减少对结果的约束，速度更快，相应的精度会降低
            as_tuple: 返回的结果是否为 (idx, head, rel) 的格式，否则返回 heads, rels

        Returns:
            依存句法树结果
        """
        if len(self.dep_vocab) == 0:
            return []
        word_attention_mask = hidden['word_cls_mask']
        result = self.model.dep_classifier.forward(
            input=hidden['word_cls_input'],
            word_attention_mask=word_attention_mask[:, 1:],
            is_processed=True
        )
        dep_arc, dep_label = result.arc_logits, result.rel_logits
        dep_arc[:, 0, 1:] = float('-inf')
        dep_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        dep_arc = dep_arc.argmax(dim=-1) if fast else eisner(dep_arc, word_attention_mask)

        dep_label = torch.argmax(dep_label, dim=-1)
        dep_label = dep_label.gather(-1, dep_arc.unsqueeze(-1)).squeeze(-1)

        dep_arc[~word_attention_mask] = -1
        dep_label[~word_attention_mask] = -1

        head_pred = [
            [item for item in arcs if item != -1]
            for arcs in dep_arc[:, 1:].cpu().numpy().tolist()
        ]
        rel_pred = [
            [self.dep_vocab[item] for item in rels if item != -1]
            for rels in dep_label[:, 1:].cpu().numpy().tolist()
        ]
        if not as_tuple:
            return head_pred, rel_pred
        return [
            [(idx + 1, head, rel) for idx, (head, rel) in enumerate(zip(heads, rels))]
            for heads, rels in zip(head_pred, rel_pred)
        ]

    @no_grad
    def sdp(self, hidden: dict, mode: str = 'mix'):
        """
        语义依存图（树）
        Args:
            hidden: 分词时所得到的中间表示
            mode: ['tree', 'graph', 'mix']

        Returns:
            语义依存图（树）结果
        """
        if len(self.sdp_vocab) == 0:
            return []

        word_attention_mask = hidden['word_cls_mask']
        result = self.model.sdp_classifier(
            input=hidden['word_cls_input'],
            word_attention_mask=word_attention_mask[:, 1:],
            is_processed=True
        )
        sdp_arc, sdp_label = result.arc_logits, result.rel_logits
        sdp_arc[:, 0, 1:] = float('-inf')
        sdp_arc.diagonal(0, 1, 2).fill_(float('-inf'))  # 避免自指
        sdp_label = torch.argmax(sdp_label, dim=-1)

        if mode == 'tree':
            # 语义依存树
            sdp_arc_idx = eisner(sdp_arc, word_attention_mask).unsqueeze_(-1).expand_as(sdp_arc)
            sdp_arc_res = torch.zeros_like(sdp_arc, dtype=torch.bool).scatter_(-1, sdp_arc_idx, True)
        elif mode == 'mix':
            # 混合解码
            sdp_arc_idx = eisner(sdp_arc, word_attention_mask).unsqueeze_(-1).expand_as(sdp_arc)
            sdp_arc_res = (sdp_arc.sigmoid_() > 0.5).scatter_(-1, sdp_arc_idx, True)
        else:
            # 语义依存图
            sdp_arc_res = torch.sigmoid_(sdp_arc) > 0.5

        sdp_arc_res[~word_attention_mask] = False
        sdp_label = get_graph_entities(sdp_arc_res, sdp_label, self.sdp_vocab)

        return sdp_label
