#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
import math
import time
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ltp.utils.task_info import TaskInfo
from ltp.utils.method_builder import default_build_method

warnings.filterwarnings("ignore")


def common_train(args, model_class, task_info: TaskInfo, build_method=default_build_method, model_kwargs: dict = None):
    if model_kwargs is None:
        model_kwargs = {}

    pl.seed_everything(args.seed)

    early_stop_callback = EarlyStopping(
        monitor=f'val_{task_info.metric_name}', min_delta=1e-5, patience=args.patience, verbose=False, mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_{task_info.metric_name}', save_top_k=1, verbose=True, mode='max', save_last=True
    )
    model = model_class(args, **model_kwargs)
    build_method(model, task_info)
    this_time = time.strftime("%m-%d_%H:%M:%S", time.localtime())

    try:
        import wandb
        logger = loggers.WandbLogger(
            project=args.project,
            save_dir='lightning_logs',
            name=f'{task_info.task_name}_{this_time}',
            offline=args.offline
        )
    except Exception as e:
        logger = loggers.TensorBoardLogger(
            save_dir='lightning_logs', name=f'{task_info.task_name}_{this_time}', default_hp_metric=False
        )
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[early_stop_callback],
        checkpoint_callback=checkpoint_callback
    )
    # Ready to train with new learning rate
    trainer.fit(model)
    trainer.test()


try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from pytorch_lightning.utilities.cloud_io import load as pl_load
    from ltp.plugins.tune import TuneReportCheckpointCallback


    def tune_train_once(
            config,
            checkpoint_dir=None,
            args: argparse.Namespace = None,
            model_class: type = None,
            build_method=None,
            task_info: TaskInfo = None,
            model_kwargs: dict = None,
            resume: str = None,
            **kwargs
    ):
        if resume is None:
            resume = 'all'
        args_vars = vars(args)
        args_vars.update(config)

        pl.seed_everything(args.seed)
        logger = [
            loggers.CSVLogger(save_dir=tune.get_trial_dir(), name="", version="."),
            loggers.TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False)
        ]
        trainer_args = dict(
            logger=logger,
            progress_bar_refresh_rate=0,
            callbacks=[
                TuneReportCheckpointCallback(
                    metrics={f'tune_{task_info.metric_name}': f'val_{task_info.metric_name}'},
                    filename="tune.ckpt", on="validation_end"
                )
            ]
        )
        if checkpoint_dir and resume == 'all':
            trainer_args['resume_from_checkpoint'] = os.path.join(checkpoint_dir, "tune.ckpt")

        # fix slurm trainer
        os.environ["SLURM_JOB_NAME"] = "bash"
        model = model_class(args, **model_kwargs)
        build_method(model, task_info)
        trainer: Trainer = Trainer.from_argparse_args(args, **trainer_args)
        if checkpoint_dir and resume == 'model':
            ckpt = pl_load(os.path.join(checkpoint_dir, "tune.ckpt"), map_location=lambda storage, loc: storage)
            model = model._load_model_state(ckpt)
            trainer.current_epoch = ckpt["epoch"]
        trainer.fit(model)


    def tune_train(args, model_class, task_info: TaskInfo, build_method=default_build_method,
                   model_kwargs: dict = None):
        if model_kwargs is None:
            model_kwargs = {}
        args.data_dir = os.path.abspath(args.data_dir)
        this_time = time.strftime("%m-%d_%H:%M:%S", time.localtime())

        lr_quantity = 10 ** round(math.log(args.lr, 10))
        config = {
            "seed": tune.choice(list(range(10))),

            # 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
            "lr": tune.uniform(lr_quantity, 5 * lr_quantity),

            # -1 for disable, 0.8 for Base/Small, 0.9 for Large
            "layerwise_lr_decay_power": tune.choice([0.8, 0.9]),

            # lr scheduler
            "lr_scheduler": tune.choice(['linear_schedule_with_warmup', 'polynomial_decay_schedule_with_warmup'])
        }
        if torch.cuda.is_available():
            resources_per_trial = {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
        else:
            resources_per_trial = {"cpu": args.cpus_per_trial}
        print("resources_per_trial", resources_per_trial)

        analysis = tune.run(
            tune.with_parameters(
                tune_train_once,
                args=args,
                task_info=task_info,
                model_class=model_class,
                build_method=build_method,
                model_kwargs=model_kwargs,
                resume='all'
            ),
            mode="max",
            config=config,
            num_samples=args.num_samples,
            metric=f'tune_{task_info.metric_name}',
            name=f'{task_info.task_name}_{this_time}',
            local_dir='lightning_logs',
            progress_reporter=CLIReporter(
                parameter_columns=list(config.keys()),
                metric_columns=[
                    "loss", f'tune_{task_info.metric_name}', "training_iteration"
                ]
            ),
            resources_per_trial=resources_per_trial,
            scheduler=ASHAScheduler(
                max_t=args.max_epochs,
                grace_period=args.min_epochs
            ),
            queue_trials=True,
            keep_checkpoints_num=3,
            checkpoint_score_attr=f'tune_{task_info.metric_name}'
        )
        print("Best hyperparameters found were: ", analysis.best_config)

        args_vars = vars(args)
        args_vars.update(analysis.best_config)
        model = model_class.load_from_checkpoint(
            os.path.join(analysis.best_checkpoint, "tune.ckpt"), hparams=args, **model_kwargs
        )
        logger = loggers.TensorBoardLogger(
            save_dir=analysis.best_trial.logdir, name="", version=".", default_hp_metric=False
        )
        trainer: Trainer = Trainer.from_argparse_args(args, logger=logger)
        build_method(model, task_info)
        trainer.test(model)

except Exception as e:
    def tune_train(*args, **kwargs):
        print("please install ray[tune]")
