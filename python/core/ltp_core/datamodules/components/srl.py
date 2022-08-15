#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import logging
import os
from collections import Counter
from dataclasses import dataclass
from os.path import join

import datasets

from ltp_core.datamodules.utils.iterator import iter_blocks
from ltp_core.datamodules.utils.vocab_helper import vocab_builder

_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"


@vocab_builder
def build_vocabs(data_dir, *files):
    counters = {"predicate": (1, Counter()), "arguments": (slice(2, None), Counter())}

    if any([os.path.exists(os.path.join(data_dir, "vocabs", f"{key}.txt")) for key in counters]):
        return

    if not os.path.exists(os.path.join(data_dir, "vocabs")):
        os.makedirs(os.path.join(data_dir, "vocabs"))

    for filename in files:
        for line_num, block in iter_blocks(filename=filename):
            values = [list(value) for value in zip(*block)]

            for name, (row, counter) in counters.items():
                current = values[row]
                if not len(current):
                    continue
                item = current[0]
                if isinstance(item, list):
                    for item in current:
                        counter.update(item)
                else:
                    counter.update(current)

    for feat, (row, counter) in counters.items():
        with open(os.path.join(data_dir, "vocabs", f"{feat}.txt"), mode="w") as f:
            # some process
            if feat == "predicate":
                tags = sorted(counter.keys())
                tags.remove("_")
                tags = ["_"] + tags
            elif feat == "arguments":
                tags = sorted(counter.keys())
                tags.remove("O")
                if "B-V" in tags:
                    tags.remove("B-V")
                    tags_backup = ["O", "B-V"]
                else:
                    tags_backup = ["O"]
                tags = sorted({tag[2:] for tag in tags})
                tags = [f"B-{tag}" for tag in tags] + [f"I-{tag}" for tag in tags]

                tags = tags_backup + tags
            else:
                tags = ["_"]
            f.write("\n".join(tags))


def create_feature(file=None):
    if file:
        return datasets.ClassLabel(names_file=file)
    return datasets.Value("string")


@dataclass
class SrlConfig(datasets.BuilderConfig):
    """BuilderConfig for Conll2003."""

    predicate: str = None
    arguments: str = None

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


# ["word", "bio"]
class Srl(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = SrlConfig

    def _info(self):
        build_vocabs(self.config)
        feats = {"predicate": self.config.predicate, "arguments": self.config.arguments}
        for key in feats:
            if feats[key] is None:
                feats[key] = os.path.join(self.config.data_dir, "vocabs", f"{key}.txt")
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "form": datasets.Sequence(datasets.Value("string")),
                    "predicate": datasets.Sequence(create_feature(feats["predicate"])),
                    "arguments": datasets.Sequence(
                        datasets.Sequence(create_feature(feats["arguments"]))
                    ),
                }
            ),
            supervised_keys=None,
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
                words, predicate, *roles = (list(value) for value in zip(*block))

                yield line_num, {
                    "form": words,
                    "predicate": predicate,
                    "arguments": roles,
                }


def main():
    from ltp_core.datamodules.utils.datasets import load_dataset

    dataset = load_dataset(Srl, data_dir="data/srl")
    print(dataset)


if __name__ == "__main__":
    main()
