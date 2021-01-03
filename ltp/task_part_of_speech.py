#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import numpy
import torch
import torch.utils.data
import os
from tqdm import tqdm
from argparse import ArgumentParser
from ltp.data import dataset as datasets
from ltp.data.utils import collate
from ltp.transformer_linear import TransformerLinear as Model
from pytorch_lightning import Trainer

from transformers import AutoTokenizer
from ltp import optimization
from ltp.utils import TaskInfo, common_train, map2device, convert2npy, tune_train, dataset_cache_wrapper

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ltp/task_part_of_speech.py --data_dir=data/pos --num_labels=27 --max_epochs=10 --batch_size=16 --gpus=1 --precision=16 --auto_lr_find=lr
@dataset_cache_wrapper()
def build_dataset(model: Model, data_dir, task_name):
    dataset = datasets.load_dataset(
        datasets.Conllu,
        data_dir=data_dir,
        cache_dir=data_dir
    )
    dataset.remove_columns_(["id", "lemma", "upos", "feats", "head", "deprel", "deps", "misc"])
    dataset.rename_column_('xpos', 'labels')

    tokenizer = AutoTokenizer.from_pretrained(model.hparams.transformer, use_fast=True)

    def tokenize(examples):
        res = tokenizer(
            examples['form'],
            is_split_into_words=True,
            max_length=model.transformer.config.max_position_embeddings,
            truncation=True
        )

        labels = []
        logits_mask = []
        for encoding, labels_ in zip(res.encodings, examples['labels']):
            labels.append([labels_[word_idx] for word_idx in encoding.words[1:-1]])

            logits_mask.append([])
            last_word_idx = -1
            for word_idx in encoding.words[1:-1]:
                if word_idx != last_word_idx:
                    logits_mask[-1].append(True)
                else:
                    logits_mask[-1].append(False)
                last_word_idx = word_idx

        res['labels'] = labels
        res['logits_mask'] = logits_mask
        return res

    dataset = dataset.map(
        lambda examples: tokenize(examples), batched=True,
        cache_file_names={
            k: d._get_cache_file_path(f"{task_name}-{k}-tokenized") for k, d in dataset.items()
        }
    )
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'logits_mask', 'labels'])
    dataset.shuffle(indices_cache_file_names={
        k: d._get_cache_file_path(f"{task_name}-{k}-shuffled-index-{model.hparams.seed}") for k, d in
        dataset.items()
    })
    return dataset


def validation_method(metric, loss_tag='val_loss', metric_tag=f'val_acc', ret=False):
    def step(self: Model, batch, batch_nb):
        result = self.forward(**batch)

        mask = batch['logits_mask']
        labels = batch['labels']
        preds = torch.argmax(result.logits, dim=-1)
        preds_true = preds[mask] == labels[mask]
        return {
            loss_tag: result.loss.item(),
            f'{metric_tag}/true': torch.sum(preds_true, dtype=torch.float).item(),
            f'{metric_tag}/all': preds_true.numel()
        }

    def epoch_end(self: Model, outputs):
        if isinstance(outputs, dict):
            outputs = [outputs]
        length = len(outputs)
        loss = sum([output[loss_tag] for output in outputs]) / length
        acc = sum([output[f'{metric_tag}/true'] for output in outputs]) / \
              sum([output[f'{metric_tag}/all'] for output in outputs])

        self.log_dict(
            dictionary={loss_tag: loss, metric_tag: acc},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if ret:
            return acc

    return step, epoch_end


task_info = TaskInfo(
    task_name='pos',
    metric_name='acc',
    build_dataset=build_dataset,
    validation_method=validation_method
)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--project', type=str, default='ltp')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gpus_per_trial', type=float, default=1.0)
    parser.add_argument('--cpus_per_trial', type=float, default=1.0)
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

    dataset, metric = build_dataset(model, args.data_dir, task_info.task_name)
    train_dataloader = torch.utils.data.DataLoader(
        dataset[datasets.Split.TRAIN],
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=args.num_workers
    )

    output = os.path.join(args.data_dir, task_info.task_name, 'output.npz')

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
            logits = model.forward(**batch).logits
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

    # set default args
    parser.set_defaults(gradient_clip_val=1.0, min_epochs=1, max_epochs=10)
    parser.set_defaults(num_labels=27)

    args = parser.parse_args()

    if args.build_dataset:
        build_distill_dataset(args)
    elif args.tune:
        tune_train(args, model_class=Model, task_info=task_info)
    else:
        common_train(args, model_class=Model, task_info=task_info)


if __name__ == '__main__':
    main()
