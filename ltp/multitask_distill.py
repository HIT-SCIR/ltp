#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
import types
import numpy as np
from argparse import ArgumentParser

import torch
import torch.utils.data
import torch.nn.functional as F

from pytorch_lightning import Trainer

import ltp
from ltp import optimization, multitask
from ltp.data import dataset as datasets
from ltp.data.utils import collate, MultiTaskDataloader
from ltp.transformer_multitask import TransformerMultiTask as Model
from ltp.utils import TaskInfo, common_train, deploy_model, tune_train, map2device
from ltp.multitask import validation_method

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


def kd_mse_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the mse loss between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    loss = F.mse_loss(beta_logits_S, beta_logits_T)
    return loss


def flsw_temperature_scheduler_builder(beta, gamma, base_temperature=8, eps=1e-4, *args):
    '''
    adapted from arXiv:1911.07471
    '''

    def flsw_temperature_scheduler(logits_S, logits_T):
        v = logits_S.detach()
        t = logits_T.detach()
        with torch.no_grad():
            v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
            t = t / (torch.norm(t, dim=-1, keepdim=True) + eps)
            w = torch.pow((1 - (v * t).sum(dim=-1)), gamma)
            tau = base_temperature + (w.mean() - w) * beta
        return tau

    return flsw_temperature_scheduler


def distill_linear(batch, result, target, temperature_scheduler, model: Model = None, extra=None) -> torch.Tensor:
    if 'logits_mask' in batch:
        logits_mask = batch['logits_mask']
    elif 'word_attention_mask' in batch:
        logits_mask = batch['word_attention_mask']
    else:
        logits_mask = batch['attention_mask'][:, 2:]
    active_logits = result.logits[logits_mask]
    active_target_logits = target[logits_mask]

    if result.decoded is not None and extra is not None:
        start_transitions = torch.as_tensor(extra['start_transitions'], device=model.device)
        transitions = torch.as_tensor(extra['transitions'], device=model.device)
        end_transitions = torch.as_tensor(extra['end_transitions'], device=model.device)

        kd_loss = kd_mse_loss(active_logits, active_target_logits)

        crf_loss = kd_mse_loss(transitions, model.srl_classifier.crf.transitions) + \
                   kd_mse_loss(start_transitions, model.srl_classifier.crf.start_transitions) + \
                   kd_mse_loss(end_transitions, model.srl_classifier.crf.end_transitions)

        return kd_loss + crf_loss
    else:
        temperature = temperature_scheduler(active_logits, active_target_logits)
        return kd_ce_loss(active_logits, active_target_logits, temperature=temperature)


def distill_matrix_dep(batch, result, target, temperature_scheduler, model: Model = None, extra=None) -> torch.Tensor:
    head = batch['head']
    logits_mask = batch['word_attention_mask']

    # Only keep active parts of the loss
    active_heads = head[logits_mask]

    arc_logits, rel_logits = result.src_arc_logits, result.rel_logits
    target_arc_logits, target_rel_logits = target

    arc_logits = arc_logits[:, 1:, :][logits_mask]
    target_arc_logits = target_arc_logits[:, 1:, :][logits_mask]

    rel_logits = rel_logits[:, 1:, :][logits_mask][torch.arange(len(active_heads)), active_heads]
    target_rel_logits = target_rel_logits[:, 1:, :][logits_mask][torch.arange(len(active_heads)), active_heads]

    temperature = temperature_scheduler(arc_logits, target_arc_logits)
    arc_loss = kd_ce_loss(arc_logits, target_arc_logits, temperature=temperature)

    temperature = temperature_scheduler(rel_logits, target_rel_logits)
    rel_loss = kd_ce_loss(rel_logits, target_rel_logits, temperature=temperature)

    classifier = model.dep_classifier
    return 2 * ((1 - classifier.loss_interpolation) * arc_loss + classifier.loss_interpolation * rel_loss)


def distill_matrix_sdp(batch, result, target, temperature_scheduler, model: Model = None, extra=None) -> torch.Tensor:
    head = batch['head']
    logits_mask = batch['word_attention_mask']

    arc_logits, rel_logits = result.src_arc_logits, result.rel_logits
    target_arc_logits, target_rel_logits = target

    arc_logits = arc_logits[:, 1:, :][logits_mask]
    target_arc_logits = target_arc_logits[:, 1:, :][logits_mask]

    rel_logits = rel_logits[:, 1:, :][head > 0]
    target_rel_logits = target_rel_logits[:, 1:, :][head > 0]

    temperature = temperature_scheduler(arc_logits, target_arc_logits)
    arc_loss = kd_mse_loss(arc_logits, target_arc_logits, temperature=temperature)

    temperature = temperature_scheduler(rel_logits, target_rel_logits)
    rel_loss = kd_ce_loss(rel_logits, target_rel_logits, temperature=temperature)

    classifier = model.dep_classifier
    return 2 * ((1 - classifier.loss_interpolation) * arc_loss + classifier.loss_interpolation * rel_loss)


def distill_matrix_crf(batch, result, target, temperature_scheduler, model: Model = None, extra=None) -> torch.Tensor:
    if 'word_attention_mask' in batch:
        logits_mask = batch['word_attention_mask']
    else:
        logits_mask = batch['attention_mask'][:, 2:]

    logits_mask = logits_mask.unsqueeze_(-1).expand(-1, -1, logits_mask.size(1))
    logits_mask = logits_mask & logits_mask.transpose(-1, -2)
    logits_mask = logits_mask.flatten(end_dim=1)
    index = logits_mask[:, 0]
    logits_mask = logits_mask[index]

    s_rel, labels = result.rel_logits, result.labels
    t_rel = target

    active_logits = s_rel[logits_mask]
    active_target_logits = t_rel[logits_mask]

    temperature = temperature_scheduler(active_logits, active_target_logits)
    kd_loss = kd_mse_loss(active_logits, active_target_logits, temperature)

    start_transitions = torch.as_tensor(extra['start_transitions'], device=model.device)
    transitions = torch.as_tensor(extra['transitions'], device=model.device)
    end_transitions = torch.as_tensor(extra['end_transitions'], device=model.device)

    # transitions_temp = temperature_scheduler(model.srl_classifier.crf.transitions, transitions)
    # s_transitions_temp = temperature_scheduler(model.srl_classifier.crf.start_transitions, start_transitions)
    # e_transitions_temp = temperature_scheduler(model.srl_classifier.crf.end_transitions, end_transitions)

    crf_loss = kd_mse_loss(transitions, model.srl_classifier.crf.transitions) + \
               kd_mse_loss(start_transitions, model.srl_classifier.crf.start_transitions) + \
               kd_mse_loss(end_transitions, model.srl_classifier.crf.end_transitions)
    return kd_loss + crf_loss


distill_loss_map = {
    'seg': distill_linear,
    'pos': distill_linear,
    'ner': distill_linear,
    'dep': distill_matrix_dep,
    'sdp': distill_matrix_sdp,
    'srl': distill_matrix_crf,
}


def build_dataset(model, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    distill_datasets = {}
    distill_datasets_extra = {}

    for task, task_data_dir in kwargs.items():
        task_distill_path = os.path.join(task_data_dir, task, 'output.npz')
        task_distill_data = np.load(task_distill_path, allow_pickle=True)

        distill_datasets[task] = task_distill_data['data'].tolist()
        distill_datasets_extra[task] = task_distill_data.get('extra', None)
        if distill_datasets_extra[task] is not None:
            distill_datasets_extra[task] = distill_datasets_extra[task].tolist()

    datasets, metrics = multitask.build_dataset(model, **kwargs)
    return (datasets, distill_datasets, distill_datasets_extra), metrics


def build_method(model: Model, task_info: TaskInfo):
    (multi_dataset, distill_datasets, distill_datasets_extra), multi_metric = build_dataset(
        model,
        seg=model.hparams.seg_data_dir,
        pos=model.hparams.pos_data_dir,
        ner=model.hparams.ner_data_dir,
        dep=model.hparams.dep_data_dir,
        sdp=model.hparams.sdp_data_dir,
        srl=model.hparams.srl_data_dir
    )

    disable_distill = {
        'seg': model.hparams.disable_seg,
        'pos': model.hparams.disable_pos,
        'ner': model.hparams.disable_ner,
        'dep': model.hparams.disable_dep,
        'sdp': model.hparams.disable_sdp,
        'srl': model.hparams.disable_srl,
    }

    disable_distill = {task for task, disable in disable_distill.items() if disable}

    temperature_scheduler = flsw_temperature_scheduler_builder(
        beta=model.hparams.distill_beta,
        gamma=model.hparams.distill_gamma,
        base_temperature=model.hparams.temperature
    )

    def train_dataloader(self):
        multi_dataloader = {
            task: torch.utils.data.DataLoader(
                task_dataset,
                batch_size=None,
                num_workers=self.hparams.num_workers,
                pin_memory=True
            )
            for task, task_dataset in distill_datasets.items()
        }
        res = MultiTaskDataloader(tau=self.hparams.tau, **multi_dataloader)
        return res

    def training_step(self: Model, batch, batch_idx):
        task = batch['task']
        target_logits = batch.pop('logits')
        result = self(**batch)
        norm_loss = result.loss

        if task not in disable_distill:
            distill_loss = distill_loss_map[task](
                batch, result, target_logits, temperature_scheduler, model,
                extra=distill_datasets_extra[task]
            )
            distill_loss_weight = self.global_step / self.num_train_steps
            loss = distill_loss_weight * norm_loss + (1 - distill_loss_weight) * distill_loss

            self.log("distill_loss", distill_loss.item())
            self.log("norm_loss", norm_loss.item())
            self.log("loss", loss.item())
            return {"loss": loss}
        else:
            self.log("loss", norm_loss.item())
            return {"loss": norm_loss}

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
        num_epoch_steps = sum(len(dataset) for dataset in distill_datasets.values())
        num_train_steps = num_epoch_steps * self.hparams.max_epochs
        setattr(self, 'num_train_steps', num_train_steps)
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
            for task_name, task_module in multitask.task_builder.items()
        }, metric_tag=f"val_{task_info.metric_name}"
    )

    model.val_dataloader = types.MethodType(val_dataloader, model)
    model.validation_step = types.MethodType(validation_step, model)
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    test_step, test_epoch_end = task_info.validation_method(
        multi_metric, loss_tag='test_loss', metric_tags={
            task_name: f"test_{task_module.task_info.metric_name}"
            for task_name, task_module in multitask.task_builder.items()
        }, metric_tag=f"test_{task_info.metric_name}"
    )

    model.test_dataloader = types.MethodType(test_dataloader, model)
    model.test_step = types.MethodType(test_step, model)
    model.test_epoch_end = types.MethodType(test_epoch_end, model)


task_info = TaskInfo(
    task_name='distill',
    metric_name='metric_mean',
    build_dataset=build_dataset,
    validation_method=validation_method
)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--seed', type=int, default=19980524)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--project', type=str, default='ltp')

    parser.add_argument('--disable_seg', action='store_true')
    parser.add_argument('--disable_pos', action='store_true')
    parser.add_argument('--disable_ner', action='store_true')
    parser.add_argument('--disable_srl', action='store_true')
    parser.add_argument('--disable_dep', action='store_true')
    parser.add_argument('--disable_sdp', action='store_true')

    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--gpus_per_trial', type=float, default=1.0)
    parser.add_argument('--cpus_per_trial', type=float, default=5.0)
    parser.add_argument('--distill_beta', type=float, default=1.0)
    parser.add_argument('--distill_gamma', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=8.0)
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
    return parser


def main():
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
    elif args.tune:
        tune_train(args, model_class=Model, task_info=task_info, build_method=build_method)
    else:
        common_train(args, model_class=Model, task_info=task_info, build_method=build_method)


if __name__ == '__main__':
    main()
