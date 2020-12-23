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

from pytorch_lightning import Trainer
from transformers import AutoTokenizer
from ltp.utils import TaskInfo, common_train, map2device, convert2npy, tune_train, dataset_cache_wrapper

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


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


@dataset_cache_wrapper(get_graph_entities)
def build_dataset(model: Model, data_dir, task_name):
    dataset = datasets.load_dataset(
        datasets.Conllu,
        data_dir=data_dir,
        cache_dir=data_dir
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
            k: d._get_cache_file_path(f"{task_name}-{k}-tokenized") for k, d in dataset.items()
        }
    )
    dataset.set_format(type='torch', columns=[
        'input_ids', 'token_type_ids', 'attention_mask', 'word_index', 'word_attention_mask', 'head', 'labels'
    ])
    dataset.shuffle(indices_cache_file_names={
        k: d._get_cache_file_path(f"{task_name}-{k}-shuffled-index-{model.hparams.seed}") for k, d in
        dataset.items()
    })
    return dataset


def validation_method(metric_func=None, loss_tag='val_loss', metric_tag=f'val_f1', ret=False):
    def step(self: Model, batch, batch_nb):
        result = self.forward(**batch)
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

    def epoch_end(self: Model, outputs):
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

        self.log_dict(
            dictionary={metric_tag.replace('f1', 'p'): p, metric_tag.replace('f1', 'r'): r},
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log_dict(
            dictionary={loss_tag: loss, metric_tag: f},
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if ret:
            return f

    return step, epoch_end


task_info = TaskInfo(
    task_name='sdp',
    metric_name='f1',
    build_dataset=build_dataset,
    validation_method=validation_method
)


def add_task_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--seed', type=int, default=19980524)
    parser.add_argument('--patience', type=int, default=5)
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
        args.resume_from_checkpoint, hparams=args, loss_func=sdp_loss
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
            result = model.forward(**batch)
            logits = result.src_arc_logits, result.rel_logits
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
    parser.set_defaults(gradient_clip_val=1.0, min_epochs=1, max_epochs=10)
    parser.set_defaults(num_labels=56, arc_hidden_size=600, rel_hidden_size=600)

    args = parser.parse_args()

    if args.build_dataset:
        build_distill_dataset(args)
    elif args.tune:
        tune_train(args, model_class=Model, task_info=task_info, model_kwargs={'loss_func': sdp_loss})
    else:
        common_train(args, model_class=Model, task_info=task_info, model_kwargs={'loss_func': sdp_loss})


if __name__ == '__main__':
    main()
