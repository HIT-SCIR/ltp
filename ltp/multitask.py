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
    task_segmention, task_part_of_speech, task_named_entity_recognition,
    task_dependency_parsing, task_semantic_dependency_parsing, task_semantic_role_labeling
)
from ltp.data import dataset as datasets
from ltp.data.utils import collate, MultiTaskDataloader
from ltp.transformer_multitask import TransformerMultiTask as Model
from ltp.utils import TaskInfo, common_train, tune_train, map2device, convert2npy
from ltp.utils import deploy_model

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ltp/multitask.py --max_epochs=10 --batch_size=16 --gpus=1 --precision=16 --seg_data_dir=data/seg --pos_data_dir=data/pos --ner_data_dir=data/ner

task_builder = {
    task_segmention.task_info.task_name: task_segmention,
    task_part_of_speech.task_info.task_name: task_part_of_speech,
    task_named_entity_recognition.task_info.task_name: task_named_entity_recognition,
    task_dependency_parsing.task_info.task_name: task_dependency_parsing,
    task_semantic_dependency_parsing.task_info.task_name: task_semantic_dependency_parsing,
    task_semantic_role_labeling.task_info.task_name: task_semantic_role_labeling,
}


def build_dataset(model, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    datasets = OrderedDict()
    metrics = OrderedDict()

    for task, task_data_dir in kwargs.items():
        dataset, metric = task_builder[task].build_dataset(model, task_data_dir, task)
        datasets[task] = dataset
        metrics[task] = metric

    return datasets, metrics


def validation_method(metric: dict = None, loss_tag='val_loss', metric_tag: str = None, metric_tags: dict = None,
                      ret=True):
    if metric is None or metric_tags is None:
        raise NotImplemented

    task_mapper = []
    step_mapper = []
    epoch_mapper = []

    for task, task_metric in metric.items():
        task_metric_tag = metric_tags[task]
        task_step, task_epoch_end = task_builder[task].validation_method(
            task_metric, loss_tag=f'{loss_tag}/{task}', metric_tag=f'{task_metric_tag}/{task}', ret=True
        )

        task_mapper.append(task)
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
        self.log(metric_tag, metric_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
                batch_size=self.hparams.batch_size,
                collate_fn=collate,
                num_workers=self.hparams.num_workers,
                pin_memory=True
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
                batch_size=self.hparams.batch_size,
                collate_fn=collate,
                num_workers=self.hparams.num_workers,
                pin_memory=True
            ) for task, task_dataset in multi_dataset.items()
        ]

    def test_dataloader(self):
        return [
            torch.utils.data.DataLoader(
                task_dataset[datasets.Split.TEST],
                batch_size=self.hparams.batch_size,
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
        multi_metric, loss_tag='val_loss', metric_tags={
            task_name: f"val_{task_module.task_info.metric_name}"
            for task_name, task_module in task_builder.items()
        }, metric_tag=f"val_{task_info.metric_name}"
    )

    model.val_dataloader = types.MethodType(val_dataloader, model)
    model.validation_step = types.MethodType(validation_step, model)
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    test_step, test_epoch_end = task_info.validation_method(
        multi_metric, loss_tag='test_loss', metric_tags={
            task_name: f"test_{task_module.task_info.metric_name}"
            for task_name, task_module in task_builder.items()
        }, metric_tag=f"test_{task_info.metric_name}"
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

    dataset, metric = task_named_entity_recognition.build_dataset(model, args.ner_data_dir, task_info.task_name)
    train_dataloader = torch.utils.data.DataLoader(
        dataset[datasets.Split.TRAIN],
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=args.num_workers
    )

    output = os.path.join(args.ner_data_dir, task_info.task_name, 'output.npz')

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
    parser.add_argument('--seed', type=int, default=19980524)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpus_per_trial', type=float, default=1.0)
    parser.add_argument('--cpus_per_trial', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--ltp_model', type=str, default=None)
    parser.add_argument('--ltp_version', type=str, default=ltp.__version__)
    parser.add_argument('--seg_data_dir', type=str, default=None)
    parser.add_argument('--pos_data_dir', type=str, default=None)
    parser.add_argument('--ner_data_dir', type=str, default=None)
    parser.add_argument('--dep_data_dir', type=str, default=None)
    parser.add_argument('--sdp_data_dir', type=str, default=None)
    parser.add_argument('--srl_data_dir', type=str, default=None)
    parser.add_argument('--build_ner_dataset', action='store_true')
    return parser


def main():
    # 如果要输出 LTP master 分支可以使用的模型，传入 ltp_adapter 参数为输出文件夹路径，如 ltp_model
    parser = ArgumentParser()
    parser = add_task_specific_args(parser)
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
        tune_train(args, model_class=Model, task_info=task_info, build_method=build_method)
    else:
        common_train(args, model_class=Model, task_info=task_info, build_method=build_method)


if __name__ == '__main__':
    main()
