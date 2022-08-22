#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import json
import os
from typing import List, Union

from ltp.generic import LTPOutput
from ltp.mixin import VOCAB_NAME, ModelHubMixin
from ltp_extension.algorithms import Hook
from ltp_extension.perceptron import Model


class LTP(ModelHubMixin):
    def __init__(self, cws: str = None, pos: str = None, ner: str = None):
        self.cws_model = Model.load(cws) if cws else None
        self.pos_model = Model.load(pos) if pos else None
        self.ner_model = Model.load(ner) if ner else None
        self.hook = Hook()

        self.supported_tasks = set()
        self._check()

    def add_word(self, word: str, freq: int = 1):
        self.hook.add_word(word, freq)

    def _check(self):
        for model, task in (
            (self.cws_model, "cws"),
            (self.pos_model, "pos"),
            (self.ner_model, "ner"),
        ):
            if model is not None:
                self.supported_tasks.add(task)

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def pipeline(self, *args, tasks: List[str] = None, threads: int = 8, return_dict: bool = True):
        if tasks is None:
            tasks = ["cws", "pos", "ner"]
        if not self.supported_tasks.issuperset(tasks):
            raise ValueError(f"Unsupported tasks: {tasks}")

        # cws, pos, ner = None, None, None
        result = {}
        for task in ["cws", "pos", "ner"]:
            if task not in tasks:
                continue
            if task == "cws":
                args = (self.cws_model(*args, threads=threads),)
                if len(self.hook):
                    args = (self.auto_hook(*args),)
            elif task == "pos":
                args = (*args, self.pos_model(*args, threads=threads))
            elif task == "ner":
                args = (*args, self.ner_model(*args, threads=threads))
            else:
                raise ValueError(f"Invalid task: {task}")
            result[task] = args[-1]

        if return_dict:
            return LTPOutput(**result)
        else:
            return LTPOutput(**result).to_tuple()

    def auto_hook(self, words: Union[List[str], List[List[str]]]):
        if isinstance(words[0], list):
            # is batch mode
            return [self.hook.hook("".join(s), s) for s in words]
        else:
            return self.hook.hook("".join(words), words)

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
        **model_kwargs,
    ):
        """Overwrite this method in case you wish to initialize your model in a different way."""

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_files = {
                task: os.path.join(model_id, model_file)
                for task, model_file in model_kwargs["config"]["tasks"].items()
            }
        else:
            model_files = {
                task: cls.download(
                    repo_id=model_id,
                    filename=model_file,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    local_files_only=local_files_only,
                )
                for task, model_file in model_kwargs["config"]["tasks"].items()
            }

        ltp = cls(**model_files)

        return ltp


def main():
    ltp = LTP.from_pretrained("models/legacy")
    ltp.add_word("姆去拿", 2)
    words, pos, ner = ltp.pipeline("他叫汤姆去拿外衣。").to_tuple()
    print(words, pos, ner)
    pos, ner = ltp.pipeline(words, tasks=["pos", "ner"]).to_tuple()
    print(pos, ner)
    (ner,) = ltp.pipeline(words, pos, tasks=["ner"]).to_tuple()
    print(ner)

    words, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。", "台湾是中国领土不可分割的一部分。"]).to_tuple()
    print(words, pos, ner)
    pos, ner = ltp.pipeline(words, tasks=["pos", "ner"]).to_tuple()
    print(pos, ner)
    (ner,) = ltp.pipeline(words, pos, tasks=["ner"]).to_tuple()
    print(ner)


if __name__ == "__main__":
    main()
