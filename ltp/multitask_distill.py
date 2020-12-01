import os
import numpy
import types
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
from ltp.utils import TaskInfo, common_train, deploy_model
from ltp.multitask import validation_method

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

task_info = TaskInfo(task_name='distill', metric_name='metric_mean')


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


def sdp_arc_loss_func(logits_S, logits_T, temperature=1):
    logits_S = torch.sigmoid(logits_S)
    logits_T = torch.sigmoid(logits_T)
    return kd_mse_loss(logits_S, logits_T, temperature=temperature)


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


def distill_linear(batch, logits, target, temperature_scheduler, model: Model = None) -> torch.Tensor:
    if 'logits_mask' in batch:
        logits_mask = batch['logits_mask']
    elif 'word_attention_mask' in batch:
        logits_mask = batch['word_attention_mask']
    else:
        logits_mask = batch['attention_mask'][:, 2:]
    active_logits = logits[logits_mask]
    active_target_logits = target[logits_mask]
    temperature = temperature_scheduler(active_logits, active_target_logits)
    return kd_ce_loss(active_logits, active_target_logits, temperature=temperature)


def build_distill_matrix(classifier_name: str, arc_loss_func=kd_ce_loss, rel_loss_func=kd_ce_loss):
    def distill_matrix(batch, logits, target, temperature_scheduler, model: Model = None) -> torch.Tensor:
        active_arc_logits, active_rel_logits = logits
        active_target_arc_logits, active_target_rel_logits = target

        temperature = temperature_scheduler(active_arc_logits, active_target_arc_logits)
        arc_loss = arc_loss_func(active_arc_logits, active_target_arc_logits, temperature=temperature)

        temperature = temperature_scheduler(active_rel_logits, active_target_rel_logits)
        rel_loss = rel_loss_func(active_rel_logits, active_target_rel_logits, temperature=temperature)

        classifier = getattr(model, classifier_name)
        return 2 * ((1 - classifier.loss_interpolation) * arc_loss + classifier.loss_interpolation * rel_loss)

    return distill_matrix


def distill_matrix_crf(batch, logits, target, temperature_scheduler, model: Model = None) -> torch.Tensor:
    if 'word_attention_mask' in batch:
        logits_mask = batch['word_attention_mask']
    else:
        logits_mask = batch['attention_mask'][:, 2:]

    logits_mask = logits_mask.unsqueeze_(-1).expand(-1, -1, logits_mask.size(1))
    logits_mask = logits_mask & logits_mask.transpose(-1, -2)
    logits_mask = logits_mask.flatten(end_dim=1)
    index = logits_mask[:, 0]
    logits_mask = logits_mask[index]

    s_rel, decoded, labels = logits
    t_rel, (start_transitions, transitions, end_transitions), labels = target

    logits_loss = kd_mse_loss(s_rel[logits_mask], t_rel[logits_mask])
    crf_loss = kd_mse_loss(transitions, model.srl_classifier.rel_crf.transitions) + \
               kd_mse_loss(start_transitions, model.srl_classifier.rel_crf.start_transitions) + \
               kd_mse_loss(end_transitions, model.srl_classifier.rel_crf.end_transitions)
    return (logits_loss + crf_loss) / 2


distill_loss_map = {
    'seg': distill_linear,
    'pos': distill_linear,
    'ner': distill_linear,
    'dep': build_distill_matrix('dep_classifier', arc_loss_func=kd_ce_loss),
    'sdp': build_distill_matrix('sdp_classifier', arc_loss_func=sdp_arc_loss_func),
    'srl': distill_matrix_crf,
}


def build_dataset(model, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    datasets, metrics = multitask.build_dataset(model, **kwargs)
    distill_datasets = {}

    for task, task_data_dir in kwargs.items():
        task_distill_path = os.path.join(task_data_dir, 'output.npz')
        task_distill_data = numpy.load(task_distill_path, allow_pickle=True)
        task_distill_data = task_distill_data['data'].tolist()

        distill_datasets[task] = task_distill_data

    return (datasets, distill_datasets), metrics


def build_method(model):
    (multi_dataset, distill_datasets), multi_metric = build_dataset(
        model,
        seg=model.hparams.seg_data_dir,
        pos=model.hparams.pos_data_dir,
        ner=model.hparams.ner_data_dir,
        dep=model.hparams.dep_data_dir,
        sdp=model.hparams.sdp_data_dir,
        srl=model.hparams.srl_data_dir
    )

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
        norm_loss, logits = self(**batch)

        distill_loss = distill_loss_map[task](batch, logits, target_logits, temperature_scheduler, model)
        distill_loss_weight = self.global_step / self.num_train_steps

        loss = distill_loss_weight * norm_loss + (1 - distill_loss_weight) * distill_loss

        self.log("distill_loss", distill_loss.item())
        self.log("norm_loss", norm_loss.item())
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
        num_epoch_steps = sum(len(dataset) for dataset in distill_datasets.values())
        num_train_steps = num_epoch_steps * self.hparams.max_epochs
        setattr(self, 'num_train_steps', num_train_steps)
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
            for task_name, task_module in multitask.task_builder
        }, metric_tag=f"val_{task_info.metric_name}"
    )

    model.val_dataloader = types.MethodType(val_dataloader, model)
    model.validation_step = types.MethodType(validation_step, model)
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    test_step, test_epoch_end = validation_method(
        multi_metric, loss_tag='test_loss', metric_tags={
            task_name: f"test_{task_module.task_info.metric_name}"
            for task_name, task_module in multitask.task_builder
        }, metric_tag=f"test_{task_info.metric_name}"
    )

    model.test_dataloader = types.MethodType(test_dataloader, model)
    model.test_step = types.MethodType(test_step, model)
    model.test_epoch_end = types.MethodType(test_epoch_end, model)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--seed', type=int, default=19980524)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--distill_beta', type=float, default=1.0)
    parser.add_argument('--distill_gamma', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=8.0)
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
