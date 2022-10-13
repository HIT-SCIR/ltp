from copy import deepcopy
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch.nn import ModuleDict
from torchmetrics import MeanMetric, MetricCollection

from ltp_core.models.ltp_model import LTPModule
from ltp_core.models.utils import instantiate_omega as instantiate


class LTPLitModule(LightningModule):
    """LightningModule for ltp_core.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model,
        metrics,
        criterions,
        layer_lrs,
        optimizer,
        scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model: LTPModule = instantiate(model)
        self.layer_lrs = instantiate(layer_lrs)
        self.optimizer = instantiate(optimizer)
        self.scheduler = instantiate(scheduler)

        # loss function
        criterions = instantiate(criterions)
        self.task_list = list(criterions.keys())
        self.criterions = ModuleDict(criterions)

        assert len(self.task_list) > 0, "No task specified"
        self.default_task = self.task_list[0]

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        metrics = instantiate(metrics)

        # must use module dict
        metrics = ModuleDict(
            {task: MetricCollection(metric, prefix=f"{task}/") for task, metric in metrics.items()}
        )
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        self.test_metrics = deepcopy(metrics)

        self.mean_metrics = MeanMetric()

    def forward(
        self,
        task_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        word_index: torch.Tensor = None,
        word_attention_mask: torch.Tensor = None,
    ):
        return self.model(
            task_name,
            input_ids,
            attention_mask,
            token_type_ids,
            word_index,
            word_attention_mask,
        )

    def step(self, task_name: str, batch: Any):
        inout_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # optional
        token_type_ids = batch.get("token_type_ids", None)

        # optional for word-level task
        word_index = batch.get("word_index", None)
        word_attention_mask = batch.get("word_attention_mask", None)

        outputs = self.forward(
            task_name,
            inout_ids,
            attention_mask,
            token_type_ids,
            word_index,
            word_attention_mask,
        )
        loss = self.criterions[task_name](outputs, **batch)
        return loss, outputs

    def training_step(self, batch: Any, batch_idx: int):
        task_name = batch.get("task_name", self.default_task)
        loss, outputs = self.step(task_name, batch)

        # log train metrics
        metric = self.train_metrics[task_name](outputs, **batch)
        self.log(
            f"train/{task_name}/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        for key, value in metric.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=False, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        [metric.reset() for metric in self.train_metrics.values()]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        task_name = self.task_list[dataloader_idx]
        loss, outputs = self.step(task_name, batch)

        # log val metrics
        self.val_metrics[task_name].update(outputs, **batch)
        self.log(
            f"val/{task_name}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )

    def validation_epoch_end(self, outputs: List[Any]):
        for metric in self.val_metrics.values():
            metric_dict = metric.compute()
            for key, value in metric_dict.items():
                self.log(
                    f"val/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    add_dataloader_idx=False,
                )
                self.mean_metrics.update(value)
            metric.reset()
        self.log(
            "val/mean_metric",
            self.mean_metrics.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.mean_metrics.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        task_name = self.task_list[dataloader_idx]
        loss, outputs = self.step(task_name, batch)

        # log test metrics
        self.test_metrics[task_name].update(outputs, **batch)
        self.log(
            f"test/{task_name}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

    def test_epoch_end(self, outputs: List[Any]):
        for metric in self.test_metrics.values():
            metric_dict = metric.compute()
            for key, value in metric_dict.items():
                self.log(
                    f"test/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    add_dataloader_idx=False,
                )
            metric.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        parameters = self.layer_lrs(self.model.named_parameters())
        optimizer = self.optimizer(parameters)
        return self.scheduler(optimizer, self.num_training_steps)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )
        num_devices = max(1, self.trainer.num_devices)
        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


def main():
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "model.yaml")
    model: LTPLitModule = hydra.utils.instantiate(cfg)
    print(model)

    target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
    preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])

    metric = model.train_metrics["pos"](preds, target)
    print(metric)

    model.on_epoch_end()
    optimizer = model.configure_optimizers()


if __name__ == "__main__":
    main()
