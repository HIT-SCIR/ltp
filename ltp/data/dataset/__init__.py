from .bio import Bio
from .conllu import Conllu
from .srl import Srl

from typing import Optional, Union, Dict, List
from datasets import DatasetBuilder, Features, Split, DatasetDict, Dataset


def load_dataset(
        builder_cls: type,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Union[Dict, List] = None,
        split: Optional[Union[str, Split]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Features] = None,
        save_infos: bool = False,
        **config_kwargs,
) -> Union[DatasetDict, Dataset]:
    # Instantiate the dataset builder
    builder_instance: DatasetBuilder = builder_cls(
        cache_dir=cache_dir,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        hash=hash,
        features=features,
        **config_kwargs,
    )

    # Download and prepare data
    builder_instance.download_and_prepare()

    # Build dataset for splits
    ds = builder_instance.as_dataset(split=split)
    if save_infos:
        builder_instance._save_infos()

    return ds
