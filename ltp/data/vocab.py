#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from functools import partial
import os
import logging

import torch
from tqdm import tqdm

from torchtext.vocab import Vectors
from torchtext.vocab import Vocab
from torchtext.vocab import pretrained_aliases

logger = logging.getLogger(__name__)

backup_load_vectors = Vocab.load_vectors


def load_vectors(self, vectors, **kwargs):
    if not isinstance(vectors, list):
        vectors = [vectors]
    for vector in vectors:
        if vector not in pretrained_aliases:
            pretrained_aliases[vector] = partial(TextVectors, name=vector)

    backup_load_vectors(self, vectors, **kwargs)


Vocab.load_vectors = load_vectors


class TextVectors(Vectors):

    def __init__(self, name, cache=None, unk_init=None, max_vectors=None):
        super(TextVectors, self).__init__(name=name, cache=cache, unk_init=unk_init, max_vectors=max_vectors)

    def cache(self, name, cache, url=None, max_vectors=None):
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            raise FileNotFoundError(name)

        if not os.path.isfile(path_pt):
            vectors_loaded = 0
            with open(path, 'r') as f:
                itos, vectors, dim = [], [], None

                for line in tqdm(f, total=max_vectors):
                    # Explicitly splitting on " " is important, so we don't
                    # get rid of Unicode non-breaking spaces in the vectors.
                    entries = line.rstrip().split()

                    word, entries = entries[0], entries[1:]
                    if dim is None and len(entries) > 1:
                        dim = len(entries)
                    elif len(entries) == 1:
                        logger.warning("Skipping token {} with 1-dimensional "
                                       "vector {}; likely a header".format(word, entries))
                        continue
                    elif dim != len(entries):
                        raise RuntimeError(
                            "Vector for token {} has {} dimensions, but previously "
                            "read vectors have {} dimensions. All vectors must have "
                            "the same number of dimensions.".format(word, len(entries), dim))

                    vectors.append([float(x) for x in entries])
                    vectors_loaded += 1
                    itos.append(word)

                    if vectors_loaded == max_vectors:
                        break

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).contiguous().view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)
