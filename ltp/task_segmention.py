import types

import numpy
import torch
import torch.utils.data
import os
from tqdm import tqdm
from argparse import ArgumentParser
from ltp.data import dataset as datasets
from ltp import optimization
from ltp.data.utils import collate
from seqeval.metrics import f1_score
from ltp.transformer_linear import TransformerLinear as Model

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from transformers import AutoTokenizer

from ltp.utils import TaskInfo, common_train, map2device, convert2npy

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

task_info = TaskInfo(task_name='seg', metric_name='f1')


# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ltp/task_segmention.py --data_dir=data/seg --num_labels=2 --max_epochs=10 --batch_size=16 --gpus=1 --precision=16 --auto_lr_find=lr

def build_dataset(model, data_dir):
    dataset = datasets.load_dataset(
        datasets.Conllu,
        data_dir=data_dir,
        cache_dir=data_dir
    )
    dataset.remove_columns_(["id", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"])

    tokenizer = AutoTokenizer.from_pretrained(model.hparams.transformer, use_fast=True)

    # {'B':1, 'I':0}
    def tokenize(examples):
        res = tokenizer(
            examples['form'],
            is_split_into_words=True,
            max_length=model.transformer.config.max_position_embeddings,
            truncation=True
        )
        labels = []
        for encoding in res.encodings:
            labels.append([])
            last_word_idx = -1
            for word_idx in encoding.words[1:-1]:
                labels[-1].append(int(word_idx != last_word_idx))
                last_word_idx = word_idx

        res['labels'] = labels
        return res

    dataset = dataset.map(
        lambda examples: tokenize(examples), batched=True,
        cache_file_names={
            k: d._get_cache_file_path(f"{task_info.task_name}-{k}-tokenized") for k, d in dataset.items()
        }
    )
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataset.shuffle(
        indices_cache_file_names={
            k: d._get_cache_file_path(f"{task_info.task_name}-{k}-shuffled-index-{model.hparams.seed}") for k, d in
            dataset.items()
        }
    )
    return dataset, f1_score


def validation_method(metric, loss_tag='val_loss', metric_tag=f'val_{task_info.metric_name}', log=True):
    label_mapper = ['I-W', 'B-W']

    def step(self: pl.LightningModule, batch, batch_nb):
        result = self(**batch)

        mask = batch['attention_mask'][:, 2:] != 1

        # acc
        labels = batch['labels']
        preds = torch.argmax(result.logits, dim=-1)

        labels[mask] = -1
        preds[mask] = -1

        labels = [[label_mapper[word] for word in sent if word != -1] for sent in labels.detach().cpu().numpy()]
        preds = [[label_mapper[word] for word in sent if word != -1] for sent in preds.detach().cpu().numpy()]

        return {'loss': result.loss.item(), 'pred': preds, 'labels': labels}

    def epoch_end(self: pl.LightningModule, outputs):
        if isinstance(outputs, dict):
            outputs = [outputs]
        length = len(outputs)
        loss = sum([output['loss'] for output in outputs]) / length
        preds = sum([output['pred'] for output in outputs], [])
        labels = sum([output['labels'] for output in outputs], [])

        f1 = metric(preds, labels)
        if log:
            self.log_dict(
                dictionary={loss_tag: loss, metric_tag: f1},
                on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
        else:
            return f1

    return step, epoch_end


def build_method(model):
    dataset, metric = build_dataset(model, model.hparams.data_dir)

    def train_dataloader(self):
        res = torch.utils.data.DataLoader(
            dataset[datasets.Split.TRAIN],
            batch_size=self.hparams.batch_size,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return res

    def training_step(self, batch, batch_nb):
        result = self(**batch)
        self.log("loss", result.loss.item())
        return result.loss

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset[datasets.Split.VALIDATION],
            batch_size=self.hparams.batch_size,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset[datasets.Split.TEST],
            batch_size=self.hparams.batch_size,
            collate_fn=collate,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    # AdamW + LR scheduler
    def configure_optimizers(self: Model):
        num_epoch_steps = (len(dataset[datasets.Split.TRAIN]) + self.hparams.batch_size - 1) // self.hparams.batch_size
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
            lr_scheduler=self.hparams.lr_scheduler,
            lr_scheduler_kwargs={
                'lr_end': self.hparams.lr_end,
                'power': self.hparams.lr_decay_power,
                'num_cycles': self.hparams.lr_num_cycles
            }
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    model.configure_optimizers = types.MethodType(configure_optimizers, model)

    model.train_dataloader = types.MethodType(train_dataloader, model)
    model.training_step = types.MethodType(training_step, model)
    # model.training_epoch_end = types.MethodType(training_epoch_end, model)

    validation_step, validation_epoch_end = validation_method(
        metric, loss_tag='val_loss', metric_tag=f'val_{task_info.metric_name}'
    )

    model.val_dataloader = types.MethodType(val_dataloader, model)
    model.validation_step = types.MethodType(validation_step, model)
    model.validation_epoch_end = types.MethodType(validation_epoch_end, model)

    test_step, test_epoch_end = validation_method(
        metric, loss_tag='test_loss', metric_tag=f'test_{task_info.metric_name}'
    )

    model.test_dataloader = types.MethodType(test_dataloader, model)
    model.test_step = types.MethodType(test_step, model)
    model.test_epoch_end = types.MethodType(test_epoch_end, model)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--seed', type=int, default=19980524)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--build_dataset', action='store_true')
    return parser


def build_distill_dataset(args):
    model = Model.load_from_checkpoint(
        args.resume_from_checkpoint, hparams=args
    )

    model.eval()
    model.freeze()

    dataset, metric = build_dataset(model, args.data_dir)
    train_dataloader = torch.utils.data.DataLoader(
        dataset[datasets.Split.TRAIN],
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=args.num_workers
    )
    output = os.path.join(args.data_dir, 'output.pt')

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
            loss, logits = model(**batch)
            batch.update(logits=logits)
            batchs.append(map2cpu(batch))
        numpy.savez(output, data=convert2npy(batchs))

    print("Done")


def main():
    parser = ArgumentParser()

    # add task level args
    parser = add_task_specific_args(parser)

    # add model specific args
    parser = Model.add_model_specific_args(parser)
    parser = optimization.add_optimizer_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # set task specific args
    parser.set_defaults(num_labels=2, max_epochs=10)

    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)

    if args.build_dataset:
        build_distill_dataset(args)
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
