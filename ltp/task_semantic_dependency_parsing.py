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
from ltp.transformer_biaffine import TransformerBiaffine as Model, sdp_loss

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import AutoTokenizer

from ltp.utils import TaskInfo, common_train, map2device, convert2npy

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

task_info = TaskInfo(task_name='sdp', metric_name='f1')


# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ltp/task_segmention.py --data_dir=data/seg --num_labels=2 --max_epochs=10 --batch_size=16 --gpus=1 --precision=16 --auto_lr_find=lr

def get_graph_entities(arcs, labels):
    arcs = torch.nonzero(arcs, as_tuple=False).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    res = []
    for arc in arcs:
        arc = tuple(arc)
        label = labels[arc]
        res.append((*arc, label))

    return set(res)


def build_dataset(model, data_dir):
    dataset = datasets.load_dataset(
        datasets.Conllu,
        data_dir=data_dir,
        cache_dir=data_dir,
        deps=os.path.join(data_dir, "deps_labels.txt")
    )
    dataset.remove_columns_(["id", "lemma", "upos", "xpos", "feats", "head", "deprel", "misc"])
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.transformer, use_fast=True)

    # {'B':1, 'I':0}
    def tokenize(examples):
        res = tokenizer(
            examples['form'],
            is_split_into_words=True,
            max_length=model.transformer.config.max_position_embeddings,
            truncation=True
        )
        word_index = []
        word_attention_mask = []
        for encoding in res.encodings:
            word_index.append([])
            word_attention_mask.append([])

            last_word_idx = -1
            current_length = 0
            for word_idx in encoding.words[1:-1]:
                if word_idx != last_word_idx:
                    word_index[-1].append(current_length)
                    word_attention_mask[-1].append(True)
                current_length += 1
                last_word_idx = word_idx

        res['word_index'] = word_index
        res['word_attention_mask'] = word_attention_mask

        heads = []
        labels = []
        for forms, deps in zip(examples['form'], examples['deps']):
            sentence_len = len(forms)
            heads.append([[0 for j in range(sentence_len + 1)] for i in range(sentence_len)])
            labels.append([[0 for j in range(sentence_len + 1)] for i in range(sentence_len)])
            for idx, head, rel in zip(deps['id'], deps['head'], deps['rel']):
                heads[-1][idx][head] = 1
                labels[-1][idx][head] = rel
        res['head'] = heads
        res['labels'] = labels
        for word_index, head in zip(res['word_index'], res['head']):
            assert len(word_index) == len(head)
        return res

    dataset = dataset.map(
        lambda examples: tokenize(examples), batched=True,
        cache_file_names={
            k: d._get_cache_file_path(f"{k}-tokenized") for k, d in dataset.items()
        }
    )
    dataset.set_format(type='torch', columns=[
        'input_ids', 'token_type_ids', 'attention_mask', 'word_index', 'word_attention_mask', 'head', 'labels'
    ])
    dataset.shuffle(indices_cache_file_names={
        k: d._get_cache_file_path(f"{task_info.task_name}-{k}-shuffled-index-{model.hparams.seed}") for k, d in
        dataset.items()
    })
    return dataset, get_graph_entities


def validation_method(metric_func=None, loss_tag='val_loss', metric_tag=f'val_{task_info.metric_name}', log=True):
    def step(self: pl.LightningModule, batch, batch_nb):
        result = self(**batch)
        loss = result.loss
        parc = result.arc_logits
        prel = result.rel_logits

        parc[:, 0, 1:] = float('-inf')
        parc.diagonal(0, 1, 2)[1:].fill_(float('-inf'))  # 避免自指

        parc = torch.sigmoid(parc[:, 1:, :]) > 0.5
        prel = torch.argmax(prel[:, 1:, :], dim=-1)

        predict = metric_func(parc, prel)
        real = metric_func(batch['head'], batch['labels'])

        return {
            loss_tag: loss.item(),
            f'{metric_tag}/correct': len(predict & real),
            f'{metric_tag}/pred': len(predict),
            f'{metric_tag}/true': len(real)
        }

    def epoch_end(self: pl.LightningModule, outputs):
        if isinstance(outputs, dict):
            outputs = [outputs]
        length = len(outputs)
        loss = sum([output[loss_tag] for output in outputs]) / length

        correct = sum([output[f'{metric_tag}/correct'] for output in outputs])
        pred = sum([output[f'{metric_tag}/pred'] for output in outputs])
        true = sum([output[f'{metric_tag}/true'] for output in outputs])

        p = correct / pred if pred > 0 else 0
        r = correct / true if true > 0 else 0
        f = 2 * p * r / (p + r) if (p + r > 0) else 0

        prefix, appendix = metric_tag.split('_', maxsplit=1)

        if log:
            self.log_dict(
                dictionary={loss_tag: loss, f'{prefix}_p': p, f'{prefix}_r': r, metric_tag: f},
                on_step=False, on_epoch=True, prog_bar=True, logger=True
            )
        else:
            return f

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
        args.resume_from_checkpoint, hparams=args, loss_func=sdp_loss
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

    output = os.path.join(args.data_dir, 'output.npz')

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

    # task specific default args
    parser.set_defaults(num_labels=56, max_epochs=10)
    parser.set_defaults(arc_hidden_size=600, rel_hidden_size=600)

    args = parser.parse_args()

    if args.build_dataset:
        build_distill_dataset(args)
    else:
        common_train(
            args,
            metric=f'val_{task_info.metric_name}',
            model_class=Model,
            build_method=build_method,
            task=task_info.task_name,
            loss_func=sdp_loss
        )


if __name__ == '__main__':
    main()
