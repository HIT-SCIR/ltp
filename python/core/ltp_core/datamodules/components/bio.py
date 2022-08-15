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

_TRAINING_FILE = "train.bio"
_DEV_FILE = "dev.bio"
_TEST_FILE = "test.bio"


@vocab_builder
def build_vocabs(data_dir, *files):
    counter = Counter()

    if os.path.exists(os.path.join(data_dir, "vocabs", "bio.txt")):
        return

    if not os.path.exists(os.path.join(data_dir, "vocabs")):
        os.makedirs(os.path.join(data_dir, "vocabs"))

    for filename in files:
        for line_num, block in iter_blocks(filename=filename):
            values = [list(value) for value in zip(*block)]
            counter.update(values[1])

    with open(os.path.join(data_dir, "vocabs", "bio.txt"), mode="w") as f:
        tags = sorted(counter.keys())
        tags.remove("O")
        f.write("\n".join(["O"] + tags))


def create_feature(file=None):
    if file:
        return datasets.ClassLabel(names_file=file)
    return datasets.Value("string")


@dataclass
class BioConfig(datasets.BuilderConfig):
    """BuilderConfig for Conll2003."""

    bio: str = None

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
class Bio(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = BioConfig

    def _info(self):
        build_vocabs(self.config)
        feats = {"bio": self.config.bio}
        for key in feats:
            if feats[key] is None:
                feats[key] = os.path.join(self.config.data_dir, "vocabs", f"{key}.txt")

        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "form": datasets.Sequence(datasets.Value("string")),
                    "bio": datasets.Sequence(create_feature(feats["bio"])),
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
            logging.info("⏳ Generating examples from = %s", files)
            for line_num, block in iter_blocks(filename=filename):
                # last example
                words, bio = (list(value) for value in zip(*block))

                yield line_num, {"form": words, "bio": bio}


def main():
    from ltp_core.datamodules.utils.datasets import load_dataset

    dataset = load_dataset(Bio, data_dir="data/ner")
    print(dataset)


if __name__ == "__main__":
    main()
