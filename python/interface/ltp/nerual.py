#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
from typing import Any, Dict, Iterable, List, Mapping, Union

import numpy as np
import torch
from ltp.generic import LTPOutput
from ltp.mixin import PYTORCH_WEIGHTS_NAME, ModelHubMixin
from ltp.module import BaseModule
from ltp_extension.algorithms import Hook, eisner, get_entities
from transformers import AutoTokenizer, BatchEncoding

from ltp_core.models.components.graph import GraphResult
from ltp_core.models.components.token import TokenClassifierResult
from ltp_core.models.ltp_model import LTPModule
from ltp_core.models.utils import instantiate


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


WORD_START = "B-W"
WORD_MIDDLE = "I-W"


class LTP(BaseModule, ModelHubMixin):
    model: LTPModule
    cws_vocab: List[str]
    pos_vocab: List[str]
    ner_vocab: List[str]
    dep_vocab: List[str]
    sdp_vocab: List[str]
    srl_vocab: List[str]

    def __init__(self, config=None, tokenizer=None):
        super().__init__()
        self.model = instantiate(config["model"])
        self.tokenizer = tokenizer
        self.hook = Hook()

        self.cws_vocab = config["vocabs"].get("cws", [WORD_MIDDLE, WORD_START])
        self.pos_vocab = config["vocabs"].get("pos", [])
        self.ner_vocab = config["vocabs"].get("ner", [])
        self.srl_vocab = config["vocabs"].get("srl", [])
        self.dep_vocab = config["vocabs"].get("dep", [])
        self.sdp_vocab = config["vocabs"].get("sdp", [])

        self.supported_tasks = set()
        self.post = {}
        self._check()

    def add_word(self, word: str, freq: int = 1):
        if len(word) > 0:
            self.hook.add_word(word, freq)

    def add_words(self, words: Union[str, List[str]], freq: int = 2):
        if isinstance(words, str):
            self.hook.add_word(words, freq)
        elif isinstance(words, Iterable):
            for word in words:
                self.hook.add_word(word, freq)

    def _check(self):
        self.eval()
        for vocab, task in (
            (self.cws_vocab, "cws"),
            (self.pos_vocab, "pos"),
            (self.ner_vocab, "ner"),
            (self.srl_vocab, "srl"),
            (self.dep_vocab, "dep"),
            (self.sdp_vocab, "sdp"),
            (self.sdp_vocab, "sdpg"),
        ):
            if vocab is not None and len(vocab) > 0:
                self.supported_tasks.add(task)
                self.post[task] = getattr(self, f"_{task}_post")

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.model.load_state_dict(state_dict, strict=strict)

    @property
    def version(self):
        from ltp import __version__

        return __version__

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    @no_grad
    def pipeline(
        self,
        inputs: Union[str, List[str], List[List[str]]],
        tasks: List[str] = None,
        raw_format=False,
        return_dict: bool = True,
    ):
        if tasks is None:
            tasks = ["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"]
        if not self.supported_tasks.issuperset(tasks):
            raise ValueError(f"Unsupported tasks: {tasks}")

        is_batch = True
        is_split_into_words = "cws" not in tasks

        if not is_split_into_words and isinstance(inputs, str):
            # 未分词的句子 但是不是batch的
            is_batch = False
            inputs = [inputs]
        elif is_split_into_words and isinstance(inputs[0], str):
            # 已分词的句子 但是不是batch的
            is_batch = False
            inputs = [inputs]

        tokenized = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=512,
            padding="longest",  # PaddingStrategy.LONGEST,
            truncation="longest_first",  # TruncationStrategy.LONGEST_FIRST,
            return_tensors="pt",  # TensorType.PYTORCH,
            is_split_into_words=is_split_into_words,
        )

        if is_split_into_words:
            word_index = []
            for encoding in tokenized.encodings:
                word_index.append([])
                last_word_idx = -1
                current_length = 0
                for word_idx in encoding.words[1:-1]:
                    if word_idx != last_word_idx:
                        word_index[-1].append(current_length)
                    current_length += 1
                    last_word_idx = word_idx
            word_attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor([True] * len(index), device=self.device) for index in word_index],
                batch_first=True,
            )
            word_index = torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(index, device=self.device) for index in word_index],
                batch_first=True,
            )
        else:
            word_attention_mask = None
            word_index = None

        model_kwargs = {k: v.to(self.device) for k, v in tokenized.items()}
        outputs = self.model.backbone(**model_kwargs)
        cache = {}
        hidden = {
            "outputs": outputs,
            "attention_mask": model_kwargs["attention_mask"],
            "word_index": word_index,
            "word_attention_mask": word_attention_mask,
        }

        store = {}
        for task in ["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"]:
            if task not in tasks:
                continue
            if task == "sdpg":
                cache_key = self.model.processor["sdp"]._get_name()
                if cache_key in cache:
                    hidden_state, attention_mask = cache[cache_key]
                else:
                    hidden_state, attention_mask = self.model.processor["sdp"](**hidden)
                    cache[cache_key] = (hidden_state, attention_mask)
                result = self.model.task_heads["sdp"](hidden_state, attention_mask)
                store[task] = self.post[task](result, hidden, store, inputs, tokenized)
            else:
                cache_key = self.model.processor[task]._get_name()
                if cache_key in cache:
                    hidden_state, attention_mask = cache[cache_key]
                else:
                    hidden_state, attention_mask = self.model.processor[task](**hidden)
                    cache[cache_key] = (hidden_state, attention_mask)
                result = self.model.task_heads[task](hidden_state, attention_mask)
                store[task] = self.post[task](result, hidden, store, inputs, tokenized)

            if not raw_format:
                if is_split_into_words:
                    sentences = inputs
                else:
                    sentences = store["cws"]

                if task == "ner":
                    new_store = []
                    for idx, sent in enumerate(store[task]):
                        words = sentences[idx]
                        new_store.append(
                            [
                                (tag, "".join(words[start : end + 1]))
                                for tag, start, end in get_entities(sent)
                            ]
                        )
                    store[task] = new_store
                if task == "srl":
                    new_store = []
                    for idx, sent in enumerate(store[task]):
                        words = sentences[idx]
                        new_store.append([])

                        for item, predicate in enumerate(words):
                            arguments = [
                                (tag, "".join(words[start : end + 1]))
                                for tag, start, end in get_entities(sent[item])
                            ]
                            if arguments:
                                new_store[-1].append(
                                    {"predicate": predicate, "arguments": arguments}
                                )
                    store[task] = new_store

        if is_batch:
            output = LTPOutput(**store)
        else:
            output = LTPOutput(**{task: predict[0] for task, predict in store.items()})

        if return_dict:
            return output
        else:
            return output.to_tuple()

    @no_grad
    def _cws_post(
        self,
        result: TokenClassifierResult,
        hidden: Dict[str, torch.Tensor],
        store: Dict[str, Any],
        inputs: List[str] = None,
        tokenized: BatchEncoding = None,
    ) -> LTPOutput:
        crf = result.crf
        logits = result.logits
        attention_mask = result.attention_mask

        char_idx = []
        char_pos = []
        for sentence, encodings in zip(inputs, tokenized.encodings):
            last = None
            char_idx.append([])
            char_pos.append([])
            for idx, (start, end) in enumerate(encodings.offsets[1:-1]):
                if start == 0 and end == 0:
                    break
                elif start == end:
                    continue
                elif (start, end) != last:
                    char_idx[-1].append(idx)
                    char_pos[-1].append(start)
                last = (start, end)
            char_pos[-1].append(len(sentence))

        if crf is None:
            decoded = logits.argmax(dim=-1)
            decoded = decoded.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()

            decoded = [
                [self.cws_vocab[tag] for tag, mask in zip(tags, masks) if mask]
                for tags, masks in zip(decoded, attention_mask)
            ]
        else:
            logits = torch.log_softmax(logits, dim=-1)
            decoded = crf.decode(logits, attention_mask)
            decoded = [[self.cws_vocab[tag] for tag in tags] for tags in decoded]
        entities = [get_entities([d[i] for i in idx]) for d, idx in zip(decoded, char_idx)]
        # t: tag, s: start, e: end
        entities = [[(s, e) for (t, s, e) in tse] for tse in entities]

        words = [
            [sent[pos[s] : pos[e + 1]] for s, e in sent_entities]
            for sent, pos, sent_entities in zip(inputs, char_pos, entities)
        ]

        if len(self.hook):
            words = [self.hook.hook(t, w) for t, w in zip(inputs, words)]
            words_len_cumsum = [np.cumsum([len(w) for w in s]) for s in words]

            entities = []
            for char_end, word_end in zip(char_pos, words_len_cumsum):
                entities.append([])
                length2index = {cl: idx for idx, cl in enumerate(char_end[1:])}
                for i, e in enumerate(word_end):
                    if i == 0:
                        entities[-1].append((0, length2index[e]))
                    else:
                        entities[-1].append((length2index[word_end[i - 1]] + 1, length2index[e]))

        words_idx = torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor([char[e[0]] for e in sent_entities], device=self.device)
                for char, sent_entities in zip(char_idx, entities)
            ],
            batch_first=True,
        )
        words_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor([True for e in sent_entities], device=self.device)
                for sent_entities in entities
            ],
            batch_first=True,
        )
        hidden["word_index"] = words_idx
        hidden["word_attention_mask"] = words_attention_mask
        return words

    @no_grad
    def _pos_post(
        self,
        result: TokenClassifierResult,
        hidden: Dict[str, torch.Tensor],
        store: Dict[str, Any],
        inputs: List[str] = None,
        tokenized: BatchEncoding = None,
    ) -> LTPOutput:
        crf = result.crf
        logits = result.logits
        attention_mask = result.attention_mask

        if crf is None:
            decoded = logits.argmax(dim=-1)
            decoded = decoded.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            decoded = [
                [self.pos_vocab[tag] for tag, mask in zip(tags, masks) if mask]
                for tags, masks in zip(decoded, attention_mask)
            ]
        else:
            logits = torch.log_softmax(logits, dim=-1)
            decoded = crf.decode(logits, attention_mask)
            decoded = [[self.pos_vocab[tag] for tag in tags] for tags in decoded]

        return decoded

    @no_grad
    def _ner_post(
        self,
        result: TokenClassifierResult,
        hidden: Dict[str, torch.Tensor],
        store: Dict[str, Any],
        inputs: List[str] = None,
        tokenized: BatchEncoding = None,
    ) -> LTPOutput:
        crf = result.crf
        logits = result.logits
        attention_mask = result.attention_mask

        if crf is None:
            decoded = logits.argmax(dim=-1)
            decoded = decoded.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            decoded = [
                [self.ner_vocab[tag] for tag, mask in zip(tags, masks) if mask]
                for tags, masks in zip(decoded, attention_mask)
            ]
        else:
            logits = torch.log_softmax(logits, dim=-1)
            decoded = crf.decode(logits, attention_mask)
            decoded = [[self.ner_vocab[tag] for tag in tags] for tags in decoded]

        return decoded

    @no_grad
    def _srl_post(
        self,
        result: TokenClassifierResult,
        hidden: Dict[str, torch.Tensor],
        store: Dict[str, Any],
        inputs: List[str] = None,
        tokenized: BatchEncoding = None,
    ) -> LTPOutput:
        crf = result.crf
        logits = result.logits
        attention_mask = result.attention_mask

        lengths = torch.sum(attention_mask, dim=-1)

        # to expand
        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, attention_mask.size(1))
        attention_mask = attention_mask & torch.transpose(attention_mask, -1, -2)
        attention_mask = attention_mask.flatten(end_dim=1)

        index = attention_mask[:, 0]
        attention_mask = attention_mask[index]
        logits = logits.flatten(end_dim=1)[index]

        if crf is None:
            decoded = logits.argmax(dim=-1)
            decoded = decoded.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            decoded = [
                [self.srl_vocab[tag] for tag, mask in zip(tags, masks) if mask]
                for tags, masks in zip(decoded, attention_mask)
            ]
        else:
            logits = torch.log_softmax(logits, dim=-1)
            decoded = crf.decode(logits, attention_mask)
            decoded = [[self.srl_vocab[tag] for tag in tags] for tags in decoded]

        lengths = lengths.cpu().numpy()

        res = []
        for length in lengths:
            res.append(decoded[:length])
            decoded = decoded[length:]

        return res

    @no_grad
    def _dep_post(
        self,
        result: GraphResult,
        hidden: Dict[str, torch.Tensor],
        store: Dict[str, Any],
        inputs: List[str] = None,
        tokenized: BatchEncoding = None,
    ) -> LTPOutput:
        s_arc = result.arc_logits.contiguous()
        s_rel = result.rel_logits.contiguous()
        attention_mask = result.attention_mask

        # mask root 和 对角线部分
        s_arc[:, 0, 1:] = float("-inf")
        s_arc.diagonal(0, 1, 2).fill_(float("-inf"))

        s_arc = s_arc.view(-1).cpu().numpy()
        length = torch.sum(attention_mask, dim=1).view(-1).cpu().numpy() + 1
        arcs = [sequence for sequence in eisner(s_arc.tolist(), length.tolist(), True)]
        rels = torch.argmax(s_rel[:, 1:], dim=-1).cpu().numpy()
        rels = [
            [self.dep_vocab[rels[s, t, a]] for t, a in enumerate(arc)]
            for s, arc in enumerate(arcs)
        ]

        return [{"head": arc, "label": rel} for arc, rel in zip(arcs, rels)]

    @no_grad
    def _sdp_post(
        self,
        result: GraphResult,
        hidden: Dict[str, torch.Tensor],
        store: Dict[str, Any],
        inputs: List[str] = None,
        tokenized: BatchEncoding = None,
        tree: bool = True,
    ) -> LTPOutput:
        s_arc = result.arc_logits.contiguous()
        s_rel = result.rel_logits.contiguous()
        attention_mask = result.attention_mask

        # mask padding 的部分
        activate_word_mask = torch.cat([attention_mask[:, :1], attention_mask], dim=1)
        activate_word_mask = activate_word_mask.unsqueeze(-1).expand_as(s_arc)
        activate_word_mask = activate_word_mask & activate_word_mask.transpose(-1, -2)
        s_arc = s_arc.masked_fill(~activate_word_mask, float("-inf"))

        # mask root 和 对角线部分
        s_arc[:, 0, 1:] = float("-inf")
        s_arc.diagonal(0, 1, 2).fill_(float("-inf"))

        # eisner 解码
        e_arcs = s_arc.view(-1).cpu().numpy()
        length = torch.sum(attention_mask, dim=1).view(-1).cpu().numpy() + 1
        e_arcs = [sequence for sequence in eisner(e_arcs.tolist(), length.tolist(), True)]

        if tree:
            rels = torch.argmax(s_rel[:, 1:], dim=-1).cpu().numpy()
            rels = [
                [self.sdp_vocab[rels[s, t, a]] for t, a in enumerate(arc)]
                for s, arc in enumerate(e_arcs)
            ]
            return [{"head": arc, "label": rel} for arc, rel in zip(e_arcs, rels)]

        for b, arc in enumerate(e_arcs):
            for s, t in enumerate(arc):
                s_arc[b, s + 1, t] = float("inf")

        # sdpg 解码
        arcs = torch.logical_and(s_arc > 0, s_arc > s_arc.transpose(-1, -2))

        rels = torch.argmax(s_rel, dim=-1)
        pred_entities = self.get_graph_entities(arcs, rels, self.sdp_vocab)

        return pred_entities

    def _sdpg_post(
        self,
        result: GraphResult,
        hidden: Dict[str, torch.Tensor],
        store: Dict[str, Any],
        inputs: List[str] = None,
        tokenized: BatchEncoding = None,
    ) -> LTPOutput:
        return self._sdp_post(result, hidden, store, inputs, tokenized, tree=False)

    @staticmethod
    def get_graph_entities(rarcs, rels, labels):
        sequence_num = rels.shape[0]
        arcs = torch.nonzero(rarcs, as_tuple=False).cpu().detach().numpy().tolist()
        rels = rels.cpu().detach().numpy()

        res = [[] for _ in range(sequence_num)]
        for idx, arc_s, arc_e in arcs:
            label = labels[rels[idx, arc_s, arc_e]]
            res[idx].append((arc_s, arc_e, label))

        return res

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        """Overwrite this method in case you wish to initialize your model in a different way."""
        map_location = torch.device(map_location)
        ltp = cls(**model_kwargs).to(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, config=ltp.model.backbone.config, use_fast=True
            )
        else:
            model_file = cls.download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_id,
                config=ltp.model.backbone.config,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
                use_fast=True,
            )

        ltp.tokenizer = tokenizer
        state_dict = torch.load(model_file, map_location=map_location)
        ltp.load_state_dict(state_dict, strict=strict)
        ltp.eval()

        return ltp


def main():
    ltp: LTP = LTP.from_pretrained("LTP/tiny")
    ltp.add_word("姆去拿", 2)
    words, pos, ner, srl, dep, sdp = ltp.pipeline(
        ["他叫汤姆去拿外衣。", "韓語：한국의 단오", "我"],
        tasks=["cws", "pos", "ner", "srl", "dep", "sdp"],
    ).to_tuple()
    print(words)
    print(pos)
    print(ner)
    print(srl)
    print(dep)
    print(sdp)


if __name__ == "__main__":
    main()
