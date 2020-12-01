import os
import types
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.utils.data
from pytorch_lightning import Trainer

import ltp
from ltp import (
    optimization,
    task_segmention, task_part_of_speech, task_named_entity_recognition,
    task_dependency_parsing, task_semantic_dependency_parsing, task_semantic_role_labeling
)
from ltp.data import dataset as datasets
from ltp.data.utils import collate, MultiTaskDataloader
from ltp.transformer_multitask import TransformerMultiTask as Model
from ltp.utils import TaskInfo, common_train
from ltp.utils import deploy_model

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

task_info = TaskInfo(task_name='multitask', metric_name='metric_mean')

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
        dataset, metric = task_builder[task].build_dataset(model, task_data_dir)
        datasets[task] = dataset
        metrics[task] = metric

    return datasets, metrics


def validation_method(metric: dict = None, loss_tag='val_loss', metric_tag: str = None, metric_tags: dict = None,
                      log=True):
    if metric is None or metric_tags is None:
        raise NotImplemented

    task_mapper = []
    step_mapper = []
    epoch_mapper = []
    metric_tag_mapper = []

    for task, task_metric in metric.items():
        task_metric_tag = metric_tags[task]
        task_step, task_epoch_end = task_builder[task].validation_method(
            task_metric, loss_tag=f'{loss_tag}/{task}', metric_tag=task_metric_tag, log=False
        )

        task_mapper.append(task)
        step_mapper.append(task_step)
        epoch_mapper.append(task_epoch_end)
        metric_tag_mapper.append(task_metric_tag)

    def step(self, batch, batch_idx, dataloader_idx=0):
        batch['task'] = task_mapper[dataloader_idx]
        return step_mapper[dataloader_idx](self, batch, batch_idx)

    def epoch_end(self, outputs):
        metrics = []
        for idx, task_output in enumerate(outputs):
            metric = epoch_mapper[idx](self, task_output)
            metrics.append(metric)
            self.log(
                f'{metric_tag_mapper[idx]}/{task_mapper[idx]}', metric,
                on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
        metric_mean = sum(metrics) / len(metrics)
        if log:
            self.log(metric_tag, metric_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            return metric_mean

    return step, epoch_end


def build_method(model):
    multi_dataset, multi_metric = build_dataset(
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
        loss, logits = self(**batch)
        self.log("loss", loss.item())
        return {"loss": loss}

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
        optimizer, scheduler = optimization.create_optimizer(
            self,
            lr=self.hparams.lr,
            num_train_steps=num_train_steps,
            weight_decay=self.hparams.weight_decay,
            warmup_steps=self.hparams.warmup_steps,
            warmup_proportion=self.hparams.warmup_proportion,
            layerwise_lr_decay_power=self.hparams.layerwise_lr_decay_power,
            n_transformer_layers=self.transformer.config.num_hidden_layers,
            get_layer_lrs=optimization.get_layer_lrs_with_crf,
            get_layer_lrs_kwargs={'crf_preffix': 'rel_crf'},
            lr_scheduler=optimization.get_polynomial_decay_schedule_with_warmup,
            lr_scheduler_kwargs={
                'lr_end': self.hparams.lr_end,
                'power': self.hparams.lr_decay_power
            }
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    model.configure_optimizers = types.MethodType(configure_optimizers, model)

    model.train_dataloader = types.MethodType(train_dataloader, model)
    model.training_step = types.MethodType(training_step, model)

    validation_step, validation_epoch_end = validation_method(
        multi_metric, loss_tag='val_loss', metric_tags={
            task_name: f"val_{task_module.task_info.metric_name}"
            for task_name, task_module in task_builder
        }, metric_tag=f"val_{task_info.metric_name}"
    )

    model.val_dataloader = types.MethodType(val_dataloader, model)
    model.validation_step = types.MethodType(validation_step, model)
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    test_step, test_epoch_end = validation_method(
        multi_metric, loss_tag='test_loss', metric_tags={
            task_name: f"test_{task_module.task_info.metric_name}"
            for task_name, task_module in task_builder
        }, metric_tag=f"test_{task_info.metric_name}"
    )

    model.test_dataloader = types.MethodType(test_dataloader, model)
    model.test_step = types.MethodType(test_step, model)
    model.test_epoch_end = types.MethodType(test_epoch_end, model)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--seed', type=int, default=19980524)
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
    return parser


def main():
    # 如果要输出 LTP master 分支可以使用的模型，传入 ltp_adapter 参数为输出文件夹路径，如 ltp_model
    parser = ArgumentParser()
    parser = add_task_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = optimization.add_optimizer_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gradient_clip_val=1.0)
    args = parser.parse_args()

    if args.ltp_model is not None and args.resume_from_checkpoint is not None:
        deploy_model(args, args.ltp_version)
    else:
        common_train(
            args,
            metric=f'val_{task_info.metric_name}',
            model_class=Model,
            build_method=build_method,
            task=task_info.task_name
        )


if __name__ == '__main__':
    main()
