from typing import List, Dict
from . import Dataset


class MixedDataset(Dataset, alias='mixed'):
    def __init__(self, path: List[Dict], file, fields, text='text'):
        datasets = {}
        if isinstance(file, str):
            file = [file] * len(path)
        for dataset, file in zip(path, file):
            init = {}
            name = dataset['name']
            for key, value in dataset.items():
                if key != 'name' and key != 'path':
                    init[key] = value
            datasets[name] = Dataset.from_params(init, path=dataset['path'], file=file, fields=fields)

        examples = []
        for name, dataset in datasets.items():
            for example in dataset.examples:
                setattr(example, text, (name, *getattr(example, text)))
                examples.append(example)

        super().__init__(examples, fields)
