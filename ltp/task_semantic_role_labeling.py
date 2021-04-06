#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import os
from argparse import ArgumentParser

import numpy
import torch
import torch.utils.data
from tqdm import tqdm
from pytorch_lightning import Trainer

from ltp import optimization
from ltp.data import dataset as datasets
from ltp.data.utils import collate
from ltp.metrics.metric import Seqeval
from ltp.transformer_biaffine_crf import TransformerBiaffineCRF as Model
from ltp.utils import TaskInfo, common_train, map2device, convert2npy, tune_train, dataset_cache_wrapper, \
    add_common_specific_args
from ltp.utils import add_tune_specific_args

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ltp/task_segmention.py --data_dir=data/seg --num_labels=2 --max_epochs=10 --batch_size=16 --gpus=1 --precision=16 --auto_lr_find=lr

# {'B':1, 'I':0}
def tokenize(examples, tokenizer, max_length):
    res = tokenizer(
        examples['form'],
        is_split_into_words=True,
        max_length=max_length,
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

    labels = []
    for predicates, roles in zip(examples['predicate'], examples['arguments']):
        sentence_len = len(predicates)
        labels.append(numpy.zeros((sentence_len, sentence_len), dtype=numpy.int64))

        for idx, predicate in enumerate(predicates):
            if predicate == 1:
                srl = numpy.asarray(roles.pop(0), dtype=numpy.int64)
                labels[-1][idx, :] = srl

    result = res.data
    for ids in result['input_ids']:
        ids[0] = tokenizer.cls_token_id
        ids[-1] = tokenizer.sep_token_id
    result['overflow'] = [len(encoding.overflowing) > 0 for encoding in res.encodings]
    result['word_index'] = word_index
    result['word_attention_mask'] = [[True] * len(index) for index in word_index]

    result['labels'] = labels
    return result


@dataset_cache_wrapper(
    extra_builder=lambda dataset: Seqeval(dataset[datasets.Split.TRAIN].features['arguments'].feature.feature.names)
)
def build_dataset(data_dir, task_name, tokenizer, max_length=512, **kwargs):
    dataset = datasets.load_dataset(
        datasets.Srl,
        data_dir=data_dir,
        cache_dir=data_dir,
        data_files=datasets.Srl.default_files(data_dir)
    )
    dataset = dataset.map(
        lambda examples: tokenize(examples, tokenizer, max_length), batched=True,
        cache_file_names={
            k: d._get_cache_file_path(f"{task_name}-{k}-tokenized") for k, d in dataset.items()
        }
    )
    dataset = dataset.filter(
        lambda x: not x['overflow'],
        cache_file_names={
            k: d._get_cache_file_path(f"{task_name}-{k}-filtered") for k, d in dataset.items()
        }
    )
    dataset.set_format(type='torch', columns=[
        'input_ids', 'token_type_ids', 'attention_mask', 'word_index', 'word_attention_mask', 'labels'
    ])
    return dataset


def validation_method(metric: Seqeval, task=f'seg', preffix='val', ret=False):
    def step(self: Model, batch, batch_nb):
        result = self.forward(**batch)

        sent_length = [len(sent) for sent in result.decoded]
        preds = result.decoded
        labels = [sent[:sent_length[idx]] for idx, sent in enumerate(result.labels.detach().cpu().numpy())]

        step_result = metric.step(batch, preds, labels, None)

        step_result['loss'] = result.loss.item()
        return step_result

    def epoch_end(self: Model, outputs):
        if isinstance(outputs, dict):
            outputs = [outputs]
        length = len(outputs)
        loss = sum([output['loss'] for output in outputs]) / length

        core_metric, epoch_result = metric.epoch_end(outputs)
        dictionary = {f'{task}/{preffix}_{k}': v for k, v in epoch_result.items()}
        dictionary[f'{task}/{preffix}_loss'] = loss

        self.log_dict(
            dictionary=dictionary,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if ret:
            return core_metric

    return step, epoch_end


task_info = TaskInfo(
    task_name='srl',
    metric_name='f1',
    build_dataset=build_dataset,
    validation_method=validation_method
)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
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

    dataset, metric = build_dataset(args.data_dir, task_info.task_name, model)
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
            logits = model.forward(**batch).rel_logits
            batch.update(logits=logits)
            batchs.append(map2cpu(batch))
        numpy.savez(
            output,
            data=convert2npy(batchs),
            extra=convert2npy({
                'transitions': model.classifier.crf.transitions,
                'start_transitions': model.classifier.crf.start_transitions,
                'end_transitions': model.classifier.crf.end_transitions
            })
        )

    print("Done")


def main():
    parser = ArgumentParser()

    # add task level args
    parser = add_common_specific_args(parser)
    parser = add_tune_specific_args(parser)
    parser = add_task_specific_args(parser)

    # add model specific args
    parser = Model.add_model_specific_args(parser)
    parser = optimization.add_optimizer_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(gradient_clip_val=1.0, min_epochs=1, max_epochs=10)
    parser.set_defaults(num_labels=97, max_epochs=10, lr_layers_getter='get_layer_lrs_with_crf')
    args = parser.parse_args()

    if args.build_dataset:
        build_distill_dataset(args)
    elif args.tune:
        tune_train(args, model_class=Model, task_info=task_info)
    else:
        common_train(args, model_class=Model, task_info=task_info)


if __name__ == '__main__':
    main()
