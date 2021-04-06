#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
import types
from argparse import ArgumentParser
from collections import OrderedDict

import numpy
import torch
import torch.utils.data
from pytorch_lightning import Trainer
from tqdm import tqdm

import ltp
from ltp import (
    optimization,
    task_segmention as seg,
    task_part_of_speech as pos,
    task_named_entity_recognition as ner,
    task_dependency_parsing as dep,
    task_semantic_dependency_parsing as sdp,
    task_semantic_role_labeling as srl
)
from ltp.data import dataset as datasets
from ltp.data.utils import collate, MultiTaskDataloader
from ltp.transformer_multitask import TransformerMultiTask as Model
from ltp.utils import TaskInfo, common_train, tune_train, map2device, convert2npy, add_common_specific_args
from ltp.utils import deploy_model, add_tune_specific_args

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ltp/multitask.py --max_epochs=10 --batch_size=16 --gpus=1 --precision=16 --seg_data_dir=data/seg --pos_data_dir=data/pos --ner_data_dir=data/ner

task_builder = {
    seg.task_info.task_name: seg,
    pos.task_info.task_name: pos,
    ner.task_info.task_name: ner,
    dep.task_info.task_name: dep,
    sdp.task_info.task_name: sdp,
    srl.task_info.task_name: srl,
}


def build_dataset(model, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    datasets = OrderedDict()
    metrics = OrderedDict()

    for task, task_data_dir in kwargs.items():
        dataset, metric = task_builder[task].build_dataset(
            data_dir=task_data_dir, task_name=task, model=model
        )
        datasets[task] = dataset
        metrics[task] = metric

    return datasets, metrics


def validation_method(metric: dict = None, task='multi', preffix='val', ret=True):
    if metric is None:
        raise NotImplemented

    task_mapper = []
    step_mapper = []
    epoch_mapper = []

    for sub_task_name, sub_task_metric in metric.items():
        task_step, task_epoch_end = task_builder[sub_task_name].validation_method(
            sub_task_metric, task=sub_task_name, preffix=preffix, ret=True
        )

        task_mapper.append(sub_task_name)
        step_mapper.append(task_step)
        epoch_mapper.append(task_epoch_end)

    def step(self, batch, batch_idx, dataloader_idx=0):
        batch['task'] = task_mapper[dataloader_idx]
        return step_mapper[dataloader_idx](self, batch, batch_idx)

    def epoch_end(self, outputs):
        metrics = []
        for idx, task_output in enumerate(outputs):
            metric = epoch_mapper[idx](self, task_output)
            metrics.append(metric)
        metric_mean = sum(metrics) / len(metrics)
        self.log(f'{task}/{preffix}_metric_mean', metric_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if ret:
            return metric_mean

    return step, epoch_end


def build_method(model: Model, task_info: TaskInfo):
    multi_dataset, multi_metric = task_info.build_dataset(
        model,
        seg=model.hparams.seg_data_dir,
        pos=model.hparams.pos_data_dir,
        ner=model.hparams.ner_data_dir,
        dep=model.hparams.dep_data_dir,
        sdp=model.hparams.sdp_data_dir,
        srl=model.hparams.srl_data_dir
    )

    def train_dataloader(self):
        multi_dataloader = {
            task: torch.utils.data.DataLoader(
                task_dataset[datasets.Split.TRAIN],
                batch_size=getattr(self.hparams, f'{task}_batch_size') or self.hparams.batch_size,
                collate_fn=collate,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                shuffle=True
            )
            for task, task_dataset in multi_dataset.items()
        }
        res = MultiTaskDataloader(tau=self.hparams.tau, **multi_dataloader)
        return res

    def training_step(self, batch, batch_idx):
        result = self(**batch)
        self.log("loss", result.loss.item())
        return {"loss": result.loss}

    def val_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                task_dataset[datasets.Split.VALIDATION],
                batch_size=getattr(self.hparams, f'{task}_batch_size') or self.hparams.batch_size,
                collate_fn=collate,
                num_workers=self.hparams.num_workers,
                pin_memory=True
            ) for task, task_dataset in multi_dataset.items()
        ]

    def test_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                task_dataset[datasets.Split.TEST],
                batch_size=getattr(self.hparams, f'{task}_batch_size') or self.hparams.batch_size,
                collate_fn=collate,
                num_workers=self.hparams.num_workers,
                pin_memory=True
            ) for task, task_dataset in multi_dataset.items()
        ]

    # AdamW + LR scheduler
    def configure_optimizers(self: Model):
        num_epoch_steps = sum(
            (len(dataset[datasets.Split.TRAIN]) + self.hparams.batch_size - 1) // self.hparams.batch_size
            for dataset in multi_dataset.values()
        )
        num_train_steps = num_epoch_steps * self.hparams.max_epochs
        optimizer, scheduler = optimization.from_argparse_args(
            self.hparams,
            model=self,
            num_train_steps=num_train_steps,
            n_transformer_layers=self.transformer.config.num_hidden_layers
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    model.configure_optimizers = types.MethodType(configure_optimizers, model)

    model.train_dataloader = types.MethodType(train_dataloader, model)
    model.training_step = types.MethodType(training_step, model)

    validation_step, validation_epoch_end = task_info.validation_method(
        multi_metric, task=task_info.task_name, preffix='val'
    )

    model.val_dataloader = types.MethodType(val_dataloader, model)
    model.validation_step = types.MethodType(validation_step, model)
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    test_step, test_epoch_end = task_info.validation_method(
        multi_metric, task=task_info.task_name, preffix='test'
    )

    model.test_dataloader = types.MethodType(test_dataloader, model)
    model.test_step = types.MethodType(test_step, model)
    model.test_epoch_end = types.MethodType(test_epoch_end, model)


task_info = TaskInfo(
    task_name='multitask',
    metric_name='metric_mean',
    build_dataset=build_dataset,
    validation_method=validation_method
)


def build_ner_distill_dataset(args):
    model = Model.load_from_checkpoint(
        args.resume_from_checkpoint, hparams=args
    )

    model.eval()
    model.freeze()

    dataset, metric = ner.build_dataset(
        model, args.ner_data_dir,
        ner.task_info.task_name
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset[datasets.Split.TRAIN],
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=args.num_workers
    )

    output = os.path.join(args.ner_data_dir, ner.task_info.task_name, 'output.npz')

    if torch.cuda.is_available():
        model.cuda()
        map2cpu = lambda x: map2device(x)
        map2cuda = lambda x: map2device(x, model.device)
    else:
        map2cpu = lambda x: x
        map2cuda = lambda x: x

    with torch.no_grad():
        batchs = []
        for batch in tqdm(train_dataloader):
            batch = map2cuda(batch)
            logits = model.forward(task='ner', **batch).logits
            batch.update(logits=logits)
            batchs.append(map2cpu(batch))
        try:
            numpy.savez(
                output,
                data=convert2npy(batchs),
                extra=convert2npy({
                    'transitions': model.ner_classifier.crf.transitions,
                    'start_transitions': model.ner_classifier.crf.start_transitions,
                    'end_transitions': model.ner_classifier.crf.end_transitions
                })
            )
        except Exception as e:
            numpy.savez(output, data=convert2npy(batchs))

    print("Done")


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--ltp_model', type=str, default=None)
    parser.add_argument('--ltp_version', type=str, default=ltp.__version__)
    parser.add_argument('--seg_data_dir', type=str, default=None)
    parser.add_argument('--pos_data_dir', type=str, default=None)
    parser.add_argument('--ner_data_dir', type=str, default=None)
    parser.add_argument('--dep_data_dir', type=str, default=None)
    parser.add_argument('--sdp_data_dir', type=str, default=None)
    parser.add_argument('--srl_data_dir', type=str, default=None)
    parser.add_argument('--seg_batch_size', type=int, default=None)
    parser.add_argument('--pos_batch_size', type=int, default=None)
    parser.add_argument('--ner_batch_size', type=int, default=None)
    parser.add_argument('--dep_batch_size', type=int, default=None)
    parser.add_argument('--sdp_batch_size', type=int, default=None)
    parser.add_argument('--srl_batch_size', type=int, default=None)
    parser.add_argument('--build_ner_dataset', action='store_true')
    return parser


def main():
    # 如果要输出 LTP master 分支可以使用的模型，传入 ltp_adapter 参数为输出文件夹路径，如 ltp_model
    parser = ArgumentParser()

    # add task level args
    parser = add_common_specific_args(parser)
    parser = add_tune_specific_args(parser)
    parser = add_task_specific_args(parser)

    # add model specific args
    parser = Model.add_model_specific_args(parser)
    parser = optimization.add_optimizer_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(min_epochs=1, max_epochs=10)
    parser.set_defaults(gradient_clip_val=1.0, lr_layers_getter='get_layer_lrs_with_crf')
    args = parser.parse_args()

    if args.ltp_model is not None and args.resume_from_checkpoint is not None:
        deploy_model(args, args.ltp_version)
    elif args.build_ner_dataset:
        build_ner_distill_dataset(args)
    elif args.tune:
        from ltp.utils.common_train import tune
        tune_config = {
            # 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
            "lr": tune.loguniform(args.tune_min_lr, args.tune_max_lr),

            # dataset split
            "tau": tune.choice([0.8, 0.9, 1.0]),

            # 梯度衰减
            "weight_decay": tune.choice([0.0, 0.01]),

            # 梯度裁剪
            "gradient_clip_val": tune.choice([1.0, 2.0, 3.0, 4.0, 5.0]),

            # lr scheduler
            "lr_scheduler": tune.choice([
                'linear_schedule_with_warmup',
                'polynomial_decay_schedule_with_warmup',
            ]),
        }
        tune_train(args, model_class=Model, task_info=task_info, build_method=build_method, tune_config=tune_config)
    else:
        common_train(args, model_class=Model, task_info=task_info, build_method=build_method)


if __name__ == '__main__':
    main()
