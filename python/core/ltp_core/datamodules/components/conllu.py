#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import itertools
import logging
import os
from collections import Counter
from dataclasses import dataclass
from os.path import join

import datasets

from ltp_core.datamodules.utils.iterator import iter_blocks
from ltp_core.datamodules.utils.vocab_helper import vocab_builder

_CITATION = """\
@misc{nivre2020universal,
    title={Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection},
    author={Joakim Nivre and Marie-Catherine de Marneffe and Filip Ginter and Jan Hajič and Christopher D. Manning and Sampo Pyysalo and Sebastian Schuster and Francis Tyers and Daniel Zeman},
    year={2020},
    eprint={2004.10643},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
Universal Dependencies (UD) is a framework for consistent annotation of grammar (parts of speech, morphological
features, and syntactic dependencies) across different human languages. UD is an open community effort with over
300 contributors producing more than 150 treebanks in 90 languages.
"""

_TRAINING_FILE = "train.conllu"
_DEV_FILE = "dev.conllu"
_TEST_FILE = "test.conllu"


@vocab_builder
def build_vocabs(data_dir, *files, min_freq=5):
    counters = {
        "word": (1, Counter()),
        "lemma": (2, Counter()),
        "upos": (3, Counter()),
        "xpos": (4, Counter()),
        "feats": (5, Counter()),
        "deprel": (7, Counter()),
        # FOR CHAR FEATS
        "word_char": (1, Counter()),
        # DEPS
        "deps": (8, Counter()),
    }

    if any([os.path.exists(os.path.join(data_dir, "vocabs", f"{key}.txt")) for key in counters]):
        return

    if not os.path.exists(os.path.join(data_dir, "vocabs")):
        os.makedirs(os.path.join(data_dir, "vocabs"))

    for filename in files:
        for line_num, block in iter_blocks(filename=filename):
            values = [list(value) for value in zip(*block)]

            for name, (row, counter) in counters.items():
                if "char" in name:
                    counter.update(itertools.chain(*values[row]))
                elif "deps" == name:
                    try:
                        deps = [
                            [label.split(":", maxsplit=1)[1] for label in dep.split("|")]
                            for dep in values[row]
                        ]
                        counter.update(itertools.chain(*deps))
                    except Exception as e:
                        counter.update("_")
                else:
                    counter.update(values[row])

    for feat, (row, counter) in counters.items():
        if "word" in feat:
            counter = Counter({word: count for word, count in counter.items() if count > min_freq})

        with open(os.path.join(data_dir, "vocabs", f"{feat}.txt"), mode="w") as f:
            f.write("\n".join(sorted(counter.keys())))


def create_feature(file=None):
    if file:
        return datasets.ClassLabel(names_file=file)
    return datasets.Value("string")


@dataclass
class ConlluConfig(datasets.BuilderConfig):
    """BuilderConfig for Conllu."""

    upos: str = None
    xpos: str = None
    feats: str = None
    deprel: str = None
    deps: str = None

    def __post_init__(self):
        if self.data_files is None:
            from datasets.data_files import DataFilesDict

            self.data_files = DataFilesDict(
                {
                    datasets.Split.TRAIN: join(self.data_dir, _TRAINING_FILE),
                    datasets.Split.VALIDATION: join(self.data_dir, _DEV_FILE),
                    datasets.Split.TEST: join(self.data_dir, _TEST_FILE),
                }
            )


# ["id", "from", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
class Conllu(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = ConlluConfig

    def _info(self):
        build_vocabs(self.config)
        feats = {
            "upos": self.config.upos,
            "xpos": self.config.xpos,
            "feats": self.config.feats,
            "deprel": self.config.deprel,
            "deps": self.config.deps,
        }

        for key in feats:
            if feats[key] is None:
                feats[key] = os.path.join(self.config.data_dir, "vocabs", f"{key}.txt")

        deps_rel_feature = create_feature(feats["deps"])
        if deps_rel_feature.num_classes > 1:
            self.config.deps = feats["deps"]

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Sequence(datasets.Value("int64")),
                    "form": datasets.Sequence(datasets.Value("string")),
                    "lemma": datasets.Sequence(datasets.Value("string")),
                    "upos": datasets.Sequence(create_feature(feats["upos"])),
                    "xpos": datasets.Sequence(create_feature(feats["xpos"])),
                    "feats": datasets.Sequence(datasets.Value("string")),
                    "head": datasets.Sequence(datasets.Value("int64")),
                    "deprel": datasets.Sequence(create_feature(feats["deprel"])),
                    "deps": datasets.Sequence(
                        {
                            "id": datasets.Value("int64"),
                            "head": datasets.Value("int64"),
                            "rel": deps_rel_feature,
                        }
                        if deps_rel_feature.num_classes > 1
                        else create_feature(None)
                    ),
                    "misc": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles."""
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"
            )
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})
            ]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        for filename in files:
            logging.info("⏳ Generating examples from = %s", filename)
            for line_num, block in iter_blocks(filename=filename):
                # last example
                id, words, lemma, upos, xpos, feats, head, deprel, deps, misc = (
                    list(value) for value in zip(*block)
                )
                if self.config.deps:
                    deps = [
                        [label.split(":", maxsplit=1) for label in dep.split("|")] for dep in deps
                    ]
                    deps = [
                        [{"id": depid, "head": int(label[0]), "rel": label[-1]} for label in dep]
                        for depid, dep in enumerate(deps)
                    ]
                    deps = list(itertools.chain(*deps))
                    if any([dep["head"] >= len(words) for dep in deps]):
                        continue

                yield line_num, {
                    "id": id,
                    "form": words,
                    "lemma": lemma,
                    "upos": upos,
                    "xpos": xpos,
                    "feats": feats,
                    "head": head,
                    "deprel": deprel,
                    "deps": deps,
                    "misc": misc,
                }


def main():
    from ltp_core.datamodules.utils.datasets import load_dataset

    dataset = load_dataset(Conllu, data_dir="data/dep")
    print(dataset)


if __name__ == "__main__":
    main()
