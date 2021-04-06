#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
import time
import argparse

import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ltp.utils.task_info import TaskInfo
from ltp.utils.method_builder import default_build_method

warnings.filterwarnings("ignore")


def add_common_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--project', type=str, default='ltp')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=19980524)
    return parser


def common_train(args, model_class, task_info: TaskInfo, build_method=default_build_method, model_kwargs: dict = None):
    if model_kwargs is None:
        model_kwargs = {}

    pl.seed_everything(args.seed)
    if args.test:
        try:
            model = model_class.load_from_checkpoint(
                args.resume_from_checkpoint, **model_kwargs
            )
        except AttributeError as e:
            if "_gpus_arg_default" in e.args[0]:
                from ltp.patchs import pl_1_2_patch_1_1
                patched_model_path = pl_1_2_patch_1_1(args.resume_from_checkpoint)
                model = model_class.load_from_checkpoint(
                    patched_model_path, **model_kwargs
                )
            else:
                raise e
        build_method(model, task_info)
        trainer: Trainer = Trainer.from_argparse_args(args)
        trainer.test(model)
    else:
        model = model_class(args, **model_kwargs)
        build_method(model, task_info)
        early_stop_callback = EarlyStopping(
            monitor=f'{task_info.task_name}/val_{task_info.metric_name}', min_delta=1e-5, patience=args.patience,
            verbose=False, mode='max'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor=f'{task_info.task_name}/val_{task_info.metric_name}', save_top_k=1, verbose=True, mode='max',
            save_last=True
        )
        this_time = time.strftime("%m-%d_%H:%M:%S", time.localtime())
        try:
            import wandb
            logger = loggers.WandbLogger(
                project=args.project,
                save_dir='lightning_logs',
                name=f'{task_info.task_name}_{this_time}',
                offline=args.offline
            )
        except Exception:
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
    from ray.tune.logger import DEFAULT_LOGGERS
    from ray.tune.logger import TBXLoggerCallback, CSVLoggerCallback
    from pytorch_lightning.utilities.cloud_io import load as pl_load
    from ltp.plugins.tune import TuneReportCheckpointCallback
    from os.path import expanduser


    def add_tune_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--tune', action='store_true')
        parser.add_argument('--tune_resume', type=str, default='all')
        parser.add_argument('--tune_keep_checkpoints_num', type=int, default=1)
        parser.add_argument('--tune_min_lr', type=float, default=1e-4)
        parser.add_argument('--tune_max_lr', type=float, default=1e-3)
        parser.add_argument('--tune_gpus_per_trial', type=float, default=1.0)
        parser.add_argument('--tune_cpus_per_trial', type=float, default=5.0)
        parser.add_argument('--tune_num_samples', type=int, default=10)
        return parser


    def tune_train_once(
            config,
            checkpoint_dir=None,
            args: argparse.Namespace = None,
            model_class: type = None,
            build_method=None,
            task_info: TaskInfo = None,
            model_kwargs: dict = None,
            resume: str = None,
            group: str = None,
            log_dir: str = None,
            **kwargs
    ):
        if resume is None:
            resume = 'all'
        args_vars = vars(args)
        args_vars.update(config)

        pl.seed_everything(args.seed)
        pl_loggers = [
            loggers.CSVLogger(save_dir=tune.get_trial_dir(), name="", version="."),
            loggers.TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False),
        ]

        try:
            import wandb
            pl_loggers.append(
                loggers.WandbLogger(
                    save_dir=log_dir or 'tune_lightning_logs',
                    project=args.project,
                    name=tune.get_trial_name(),
                    id=tune.get_trial_id(),
                    offline=args.offline,
                    group=group
                )
            )
        except Exception:
            pass

        trainer_args = dict(
            logger=pl_loggers,
            progress_bar_refresh_rate=0,
            callbacks=[
                TuneReportCheckpointCallback(
                    metrics={f'tune_{task_info.metric_name}': f'{task_info.task_name}/val_{task_info.metric_name}'},
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
                   model_kwargs: dict = None, tune_config=None):
        if model_kwargs is None:
            model_kwargs = {}
        this_time = time.strftime("%m-%d_%H:%M:%S", time.localtime())
        experiment_name = f'{task_info.task_name}_{this_time}'

        if tune_config is None:
            config = {
                # 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
                "lr": tune.loguniform(args.tune_min_lr, args.tune_max_lr),

                # -1 for disable, 0.8 for Base/Small, 0.9 for Large
                "layerwise_lr_decay_power": tune.choice([0.8, 0.9]),

                # lr scheduler
                "lr_scheduler": tune.choice(['linear_schedule_with_warmup', 'polynomial_decay_schedule_with_warmup']),
            }
        else:
            config = tune_config
        if torch.cuda.is_available():
            resources_per_trial = {"cpu": args.tune_cpus_per_trial, "gpu": args.tune_gpus_per_trial}
        else:
            resources_per_trial = {"cpu": args.tune_cpus_per_trial}
        print("resources_per_trial", resources_per_trial)

        tune_dir = os.path.abspath('tune_lightning_logs')

        analysis = tune.run(
            tune.with_parameters(
                tune_train_once,
                args=args,
                task_info=task_info,
                model_class=model_class,
                build_method=build_method,
                model_kwargs=model_kwargs,
                resume=args.tune_resume,
                group=experiment_name,
                log_dir=tune_dir,
            ),
            mode="max",
            config=config,
            num_samples=args.tune_num_samples,
            metric=f'tune_{task_info.metric_name}',
            name=experiment_name,
            progress_reporter=CLIReporter(
                parameter_columns=list(config.keys()),
                metric_columns=[
                    "loss", f'tune_{task_info.metric_name}', "training_iteration"
                ]
            ),
            callbacks=[
                TBXLoggerCallback(),
                CSVLoggerCallback()
            ],
            resources_per_trial=resources_per_trial,
            scheduler=ASHAScheduler(
                max_t=args.max_epochs + 1,  # for test
                grace_period=args.min_epochs
            ),
            queue_trials=True,
            keep_checkpoints_num=args.tune_keep_checkpoints_num,
            checkpoint_score_attr=f'tune_{task_info.metric_name}',
            local_dir=tune_dir,
        )
        print("Best hyperparameters found were: ", analysis.best_config)
        print("Best checkpoint: ", analysis.best_checkpoint)

        args_vars = vars(args)
        args_vars.update(analysis.best_config)
        model = model_class.load_from_checkpoint(
            os.path.join(analysis.best_checkpoint, "tune.ckpt"), hparams=args, **model_kwargs
        )

        pl_loggers = [
            loggers.CSVLogger(save_dir=tune.get_trial_dir(), name="", version="."),
            loggers.TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False),
        ]

        try:
            import wandb
            pl_loggers.append(
                loggers.WandbLogger(
                    save_dir=tune_dir,
                    project=args.project,
                    name=tune.get_trial_name(),
                    id=tune.get_trial_id(),
                    offline=args.offline,
                    group=experiment_name
                )
            )
        except Exception:
            pass

        trainer: Trainer = Trainer.from_argparse_args(args, logger=pl_loggers)
        build_method(model, task_info)
        trainer.test(model)



except Exception as e:
    def tune_train(*args, **kwargs):
        print("please install ray[tune]")


    def add_tune_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--tune', action='store_true')
        return parser
