#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
from typing import Dict, List, Optional, Union

from torch import Tensor
from pytorch_lightning import Trainer, LightningModule

from ray import tune
from ray.tune.integration import pytorch_lightning as tune_pl


class TuneReportCallback(tune_pl.TuneReportCallback):
    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        # Don't report if just doing initial validation sanity checks.
        if trainer.running_sanity_check:
            return
        if not self._metrics:
            report_dict = {
                k: v.item() if isinstance(v, Tensor) else v
                for k, v in trainer.callback_metrics.items()
            }
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                metric_value = trainer.callback_metrics[metric]
                report_dict[key] = metric_value.item() if isinstance(metric_value, Tensor) else metric_value
        tune.report(**report_dict)


class TuneCheckpointCallback(tune_pl.TuneCallback):
    def __init__(self, filename: str = "checkpoint", on: Union[str, List[str]] = "validation_end"):
        super(TuneCheckpointCallback, self).__init__(on)
        self._filename = filename

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.running_sanity_check:
            return
        with tune.checkpoint_dir(step=trainer.global_step) as checkpoint_dir:
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, self._filename))


class TuneReportCheckpointCallback(tune_pl.TuneCallback):
    def __init__(self,
                 metrics: Union[None, str, List[str], Dict[str, str]] = None,
                 filename: str = "checkpoint", on=None):
        super(TuneReportCheckpointCallback, self).__init__(on)
        if on is None:
            on = ["validation_end", "test_end"]
        self._checkpoint = TuneCheckpointCallback(filename, on)
        self._report = TuneReportCallback(metrics, on)

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        self._checkpoint._handle(trainer, pl_module)
        self._report._handle(trainer, pl_module)
