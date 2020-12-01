import logging

import itertools
import datasets
from os.path import join
from dataclasses import dataclass
from ltp.data.utils import iter_blocks

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


def create_feature(file=None):
    if file:
        return datasets.ClassLabel(names_file=file)
    return datasets.Value('string')


@dataclass
class ConlluConfig(datasets.BuilderConfig):
    """BuilderConfig for Conllu"""

    upos: str = None
    xpos: str = None
    deprel: str = None
    deps: str = None


# ["id", "from", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
class Conllu(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = ConlluConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Sequence(datasets.Value("int64")),
                    "form": datasets.Sequence(datasets.Value("string")),
                    "lemma": datasets.Sequence(datasets.Value("string")),
                    "upos": datasets.Sequence(create_feature(self.config.upos)),
                    "xpos": datasets.Sequence(create_feature(self.config.xpos)),
                    "feats": datasets.Sequence(datasets.Value("string")),
                    "head": datasets.Sequence(datasets.Value("int64")),
                    "deprel": datasets.Sequence(create_feature(self.config.deprel)),
                    "deps": datasets.Sequence(
                        {
                            'id': datasets.Value('int64'),
                            'head': datasets.Value("int64"),
                            'rel': create_feature(self.config.deps)
                        } if self.config.deps else create_feature(self.config.deps)
                    ),
                    "misc": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = {
            "train": join(self.config.data_dir, _TRAINING_FILE),
            "dev": join(self.config.data_dir, _DEV_FILE),
            "test": join(self.config.data_dir, _TEST_FILE),
        }
        data_files = dl_manager.download_and_extract(data_files)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)
        for line_num, block in iter_blocks(filename=filepath):
            # last example
            id, words, lemma, upos, xpos, feats, head, deprel, deps, misc = [list(value) for value in zip(*block)]
            if self.config.deps:
                deps = [[label.split(':', maxsplit=1) for label in dep.split('|')] for dep in deps]
                deps = [[{'id': depid, 'head': int(label[0]), 'rel': label[1]} for label in dep] for depid, dep in
                        enumerate(deps)]
                deps = list(itertools.chain(*deps))
                if any([dep['head'] >= len(words) for dep in deps]):
                    continue

            yield line_num, {
                "id": id, "form": words, "lemma": lemma, "upos": upos, "xpos": xpos,
                "feats": feats, "head": head, "deprel": deprel, "deps": deps, "misc": misc,
            }
