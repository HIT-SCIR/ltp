from typing import Any, Dict, Optional

import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ltp_core.datamodules.utils.collate import collate
from ltp_core.datamodules.utils.multitask_dataloader import MultiTaskDataloader


class MultiTaskDataModule(LightningDataModule):
    """LightningDataModule for LTP datasets.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self, tokenizer, datamodules, tau=0.8, num_workers=None, pin_memory=None):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.datamodules = {
            task: info.load(tokenizer=self.tokenizer) for task, info in datamodules.items()
        }
        self.data_train: Optional[Dict[str, Dataset]] = {
            name: dataset[datasets.Split.TRAIN] for name, dataset in self.datamodules.items()
        }
        self.data_val: Optional[Dict[str, Dataset]] = {
            name: dataset[datasets.Split.VALIDATION] for name, dataset in self.datamodules.items()
        }
        self.data_test: Optional[Dict[str, Dataset]] = {
            name: dataset[datasets.Split.TEST] for name, dataset in self.datamodules.items()
        }

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        pass

    def train_dataloader(self):
        return MultiTaskDataloader(
            tau=self.hparams.tau,
            **{
                name: DataLoader(
                    dataset=dataset,
                    collate_fn=collate,
                    batch_size=self.hparams.datamodules[name].batch_size,
                    num_workers=self.hparams.num_workers
                    or self.hparams.datamodules[name].num_workers,
                    pin_memory=self.hparams.pin_memory
                    or self.hparams.datamodules[name].pin_memory,
                    shuffle=True,
                )
                for name, dataset in self.data_train.items()
            }
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset=dataset,
                collate_fn=collate,
                batch_size=self.hparams.datamodules[name].batch_size,
                num_workers=self.hparams.num_workers or self.hparams.datamodules[name].num_workers,
                pin_memory=self.hparams.pin_memory or self.hparams.datamodules[name].pin_memory,
                shuffle=False,
            )
            for name, dataset in self.data_val.items()
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                dataset=dataset,
                collate_fn=collate,
                batch_size=self.hparams.datamodules[name].batch_size,
                num_workers=self.hparams.num_workers or self.hparams.datamodules[name].num_workers,
                pin_memory=self.hparams.pin_memory or self.hparams.datamodules[name].pin_memory,
                shuffle=False,
            )
            for name, dataset in self.data_test.items()
        ]

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "multi_task_datamodules.yaml")
    datamodule = hydra.utils.instantiate(cfg)

    val_dataloaders = datamodule.val_dataloader()
    print(val_dataloaders)
