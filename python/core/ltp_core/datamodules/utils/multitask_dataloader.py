#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import numpy as np


def cycle(iterable):
    while True:
        yield from iterable


class MultiTaskDataloader:
    def __init__(self, tau=1.0, **dataloaders):
        self.dataloaders = dataloaders

        Z = sum(pow(v, tau) for v in self.dataloader_sizes.values())
        self.tasknames, self.sampling_weights = zip(
            *((k, pow(v, tau) / Z) for k, v in self.dataloader_sizes.items())
        )
        self.dataiters = {k: cycle(v) for k, v in dataloaders.items()}

    @property
    def dataloader_sizes(self):
        if not hasattr(self, "_dataloader_sizes"):
            self._dataloader_sizes = {k: len(v) for k, v in self.dataloaders.items()}
        return self._dataloader_sizes

    def __len__(self):
        return sum(v for k, v in self.dataloader_sizes.items())

    def __iter__(self):
        for i in range(len(self)):
            taskname = np.random.choice(self.tasknames, p=self.sampling_weights)
            dataiter = self.dataiters[taskname]
            batch = next(dataiter)

            batch["task_name"] = taskname

            yield batch
