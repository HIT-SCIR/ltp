#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import torch.utils.data
import random
from contextlib import contextmanager
from copy import deepcopy
import os

from ltp import Registrable


class Dataset(torch.utils.data.dataset.Dataset, metaclass=Registrable):
    """
    数据集抽象类，在配置文件中的配置项通常为::

        [Dataset]
        class = "CTB"

        [Dataset.init]
        path="data/postag"
        train="train.txt"
        validation="dev.txt"
        test="test.txt"

    定义一个由 Examples 和 Fields 组合成的 Dataset

    Attributes:
        sort_key (callable): 用于在生成Batch的是时候进行排序的函数
        examples (List[Example]): 数据集的 Examples
        fields (dict[str, Field]): Fields，同一个 Field Object 会共享它们的 Vocab
    """
    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """使用examples和fields创建一个Dataset

        Args:
            examples: Examples的列表
            fields: List(tuple(str, Field))
            filter_pred: 只有当该函数返回值为True时，该Example才会被使用
        """

        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        self.examples = examples
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        """ 一次创建多个数据集

        Args:
            path: 数据集路径前缀
            root: 根目录
            train: 训练集文件名
            validation: 验证集文件名
            test: 测试集文件名
            **kwargs: 其他传给Dataset的参数

        Returns:
            Tuple[Dataset] train, validation, test
        """
        if path is None:
            path = cls.download(root)
        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):
        """通过切分当前数据集的Examples以获得多个数据集

        Args:
            split_ratio (float/list(float)): 切分比例(默认为0.7)
            stratified: 是否进行分层采样
            strata_field: 分层采样的field
            random_state: 用于Shuffle的随机种子`random.getstate()`的返回值

        Returns:
            Tuple[Dataset] train, validation, test Dataset
        """
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)

        # For the permutations
        rnd = RandomShuffler(random_state)
        if not stratified:
            train_data, test_data, val_data = rationed_split(self.examples, train_ratio,
                                                             test_ratio, val_ratio, rnd)
        else:
            if strata_field not in self.fields:
                raise ValueError("Invalid field name for strata_field {}"
                                 .format(strata_field))
            strata = stratify(self.examples, strata_field)
            train_data, test_data, val_data = [], [], []
            for group in strata:
                # Stratify each group and add together the indices.
                group_train, group_test, group_val = rationed_split(group, train_ratio,
                                                                    test_ratio, val_ratio,
                                                                    rnd)
                train_data += group_train
                test_data += group_test
                val_data += group_val

        splits = tuple(Dataset(d, self.fields)
                       for d in (train_data, val_data, test_data) if d)

        # In case the parent sort key isn't none
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key
        return splits

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2 ** 32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    def filter_examples(self, field_names):
        """Remove unknown words from dataset examples with respect to given field.

        Arguments:
            field_names (list(str)): Within example only the parts with field names in
                field_names will have their unknown words deleted.
        """
        for i, example in enumerate(self.examples):
            for field_name in field_names:
                vocab = set(self.fields[field_name].vocab.stoi)
                text = getattr(example, field_name)
                example_part = [word for word in text if word in vocab]
                setattr(example, field_name, example_part)
            self.examples[i] = example


def check_split_ratio(split_ratio):
    """Check that the split ratio argument is not malformed"""
    valid_ratio = 0.
    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        # Assert in bounds, validation size is zero
        assert 0. < split_ratio < 1., (
            "Split ratio {} not between 0 and 1".format(split_ratio))

        test_ratio = 1. - split_ratio
        return (split_ratio, test_ratio, valid_ratio)
    elif isinstance(split_ratio, list):
        # A list of relative ratios is provided
        length = len(split_ratio)
        assert length == 2 or length == 3, (
            "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.:
            split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError('Split ratio must be float or a list, got {}'
                         .format(type(split_ratio)))


def stratify(examples, strata_field):
    # The field has to be hashable otherwise this doesn't work
    # There's two iterations over the whole dataset here, which can be
    # reduced to just one if a dedicated method for stratified splitting is used
    unique_strata = set(getattr(example, strata_field) for example in examples)
    strata_maps = {s: [] for s in unique_strata}
    for example in examples:
        strata_maps[getattr(example, strata_field)].append(example)
    return list(strata_maps.values())


def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
    """Create a random permutation of examples, then split them by ratios

    Arguments:
        examples: a list of data
        train_ratio, test_ratio, val_ratio: split fractions.
        rnd: a random shuffler

    Examples:
        >>> import ltp
        >>> examples = []
        >>> train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
        >>> rnd = ltp.data.dataset.RandomShuffler(None)
        >>> train_examples, test_examples, valid_examples = \
                ltp.data.dataset.rationed_split(examples, train_ratio,
                                                      test_ratio, val_ratio,
                                                      rnd)
    """
    N = len(examples)
    randperm = rnd(range(N))
    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if not val_ratio:
        test_len = N - train_len
    else:
        test_len = int(round(test_ratio * N))

    indices = (randperm[:train_len],  # Train
               randperm[train_len:train_len + test_len],  # Test
               randperm[train_len + test_len:])  # Validation

    # There's a possibly empty list for the validation set
    data = tuple([examples[i] for i in index] for index in indices)

    return data


class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))
