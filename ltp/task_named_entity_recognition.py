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
from ltp import optimization
from seqeval.metrics import f1_score
from ltp.transformer_rel_linear import TransformerRelLinear as Model

from pytorch_lightning import Trainer
from transformers import AutoTokenizer
from ltp.utils import TaskInfo, common_train, map2device, convert2npy, tune_train, dataset_cache_wrapper

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ltp/task_named_entity_recognition.py --data_dir=data/ner --num_labels=13 --max_epochs=10 --batch_size=16 --gpus=1 --precision=16
@dataset_cache_wrapper(
    extra_builder=lambda dataset: (f1_score, dataset[datasets.Split.TRAIN].features['labels'].feature.names)
)
def build_dataset(model: Model, data_dir, task_name):
    dataset = datasets.load_dataset(
        datasets.Bio,
        data_dir=data_dir,
        cache_dir=data_dir,
        bio=os.path.join(data_dir, "ner_labels.txt")
    )
    dataset.rename_column_('bio', 'labels')
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.transformer, use_fast=True)

    def tokenize(examples):
        res = tokenizer(
            examples['words'],
            is_split_into_words=True,
            max_length=model.transformer.config.max_position_embeddings,
            truncation=True
        )
        word_index = []
        for encoding in res.encodings:
            word_index.append([])

            last_word_idx = -1
            current_length = 0
            for word_idx in encoding.words[1:-1]:
                if word_idx != last_word_idx:
                    word_index[-1].append(current_length)
                current_length += 1
                last_word_idx = word_idx

        res['word_index'] = word_index
        res['word_attention_mask'] = [[True] * len(index) for index in word_index]
        return res

    dataset = dataset.map(
        lambda examples: tokenize(examples), batched=True,
        cache_file_names={
            k: d._get_cache_file_path(f"{task_name}-{k}-tokenized") for k, d in dataset.items()
        }
    )
    dataset.set_format(type='torch', columns=[
        'input_ids', 'token_type_ids', 'attention_mask', 'word_index', 'word_attention_mask', 'labels'
    ])
    dataset.shuffle(
        indices_cache_file_names={
            k: d._get_cache_file_path(f"{task_name}-{k}-shuffled-index-{model.hparams.seed}") for k, d in
            dataset.items()
        })
    return dataset


def validation_method(metric, loss_tag='val_loss', metric_tag=f'val_f1', ret=False):
    metric_func, label_feature = metric

    def step(self: Model, batch, batch_nb):
        result = self.forward(**batch)

        mask = batch['word_attention_mask'] == False
        labels = batch['labels']
        labels[mask] = -1
        labels = [[label_feature[word] for word in sent if word != -1] for sent in labels.detach().cpu().numpy()]

        if result.decoded is None:
            # acc
            preds = torch.argmax(result.logits, dim=-1)
            preds[mask] = -1
            preds = [[label_feature[word] for word in sent if word != -1] for sent in preds.detach().cpu().numpy()]
        else:
            preds = [[label_feature[word] for word in sent] for sent in result.decoded]

        return {'loss': result.loss.item(), 'pred': preds, 'labels': labels}

    def epoch_end(self: Model, outputs):
        if isinstance(outputs, dict):
            outputs = [outputs]
        length = len(outputs)
        loss = sum([output['loss'] for output in outputs]) / length
        preds = sum([output['pred'] for output in outputs], [])
        labels = sum([output['labels'] for output in outputs], [])

        f1 = metric_func(preds, labels)

        self.log_dict(
            dictionary={loss_tag: loss, metric_tag: f1},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        if ret:
            return f1

    return step, epoch_end


task_info = TaskInfo(
    task_name='ner',
    metric_name='f1',
    build_dataset=build_dataset,
    validation_method=validation_method
)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=19980524)
    parser.add_argument('--gpus_per_trial', type=float, default=1.0)
    parser.add_argument('--cpus_per_trial', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=10)
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

        try:
            numpy.savez(
                output,
                data=convert2npy(batchs),
                extra=convert2npy({
                    'transitions': model.classifier.crf.transitions,
                    'start_transitions': model.classifier.crf.start_transitions,
                    'end_transitions': model.classifier.crf.end_transitions
                })
            )
        except Exception as e:
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
    parser.set_defaults(num_labels=13)

    args = parser.parse_args()

    if args.build_dataset:
        build_distill_dataset(args)
    elif args.tune:
        tune_train(args, model_class=Model, task_info=task_info)
    else:
        common_train(args, model_class=Model, task_info=task_info)


if __name__ == '__main__':
    main()
