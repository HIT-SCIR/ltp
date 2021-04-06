from typing import Tuple

import torch
from seqeval.metrics import f1_score

from ltp.algorithms import eisner


class Metric(object):
    def step(self, batch, preds, targets, mask) -> dict:
        raise NotImplementedError()

    def epoch_end(self, step_results) -> Tuple[float, dict]:
        raise NotImplementedError()


class Acc(Metric):
    def step(self, batch, preds, targets, mask) -> dict:
        preds_true = preds[mask] == targets[mask]

        return {
            'true': torch.sum(preds_true, dtype=torch.float).item(),
            'all': preds_true.numel()
        }

    def epoch_end(self, step_results) -> Tuple[float, dict]:
        acc = sum([output[f'true'] for output in step_results]) / \
              sum([output[f'all'] for output in step_results])

        return acc, {'acc': acc}


class Seqeval(Metric):
    def __init__(self, labels: list):
        self.labels = labels

    def step(self, batch, preds, targets, mask) -> dict:
        if isinstance(preds, torch.Tensor) or isinstance(targets, torch.Tensor):
            targets[mask] = -1
            preds[mask] = -1

            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()

        labels = [[self.labels[tag] for tag in sent if tag != -1] for sent in targets]
        preds = [[self.labels[tag] for tag in sent if tag != -1] for sent in preds]

        return {'pred': preds, 'labels': labels}

    def epoch_end(self, outputs) -> Tuple[float, dict]:
        preds = sum([output['pred'] for output in outputs], [])
        labels = sum([output['labels'] for output in outputs], [])

        f1 = f1_score(y_true=labels, y_pred=preds)
        return f1, {'f1': f1}


class Graph(Metric):
    def __init__(self, punct_idx):
        self.punct_idx = punct_idx

    def get_graph_entities(self, arcs, labels):
        arcs = torch.nonzero(arcs, as_tuple=False).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        res = []
        for arc in arcs:
            arc = tuple(arc)
            label = labels[arc]

            res.append((*arc, label))

        return set(res)

    def step(self, batch, preds, targets, mask) -> dict:
        parc, prel = preds
        rarc, rrel = targets

        if len(parc.shape) == 3:
            parc = torch.sigmoid(parc) > 0.5
        else:
            parc = torch.argmax(parc, dim=-1)

        # cat cls
        # 去除无效位置
        mask = torch.cat([mask[:, :1], mask], dim=1)
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        mask[:, 0] = 0
        mask.diagonal(0, 1, 2).fill_(0)  # 避免自指
        parc[~mask] = 0

        parc = parc[:, 1:, :]
        prel = torch.argmax(prel[:, 1:, :], dim=-1)

        # 对 punc 不计算分数
        punc_mask = rrel == self.punct_idx

        parc[punc_mask] = 0
        rarc[punc_mask] = 0

        predict = self.get_graph_entities(parc, prel)
        real = self.get_graph_entities(rarc, rrel)

        return {
            'correct': len(predict & real),
            'pred': len(predict),
            'true': len(real)
        }

    def epoch_end(self, outputs) -> Tuple[float, dict]:
        correct = sum([output[f'correct'] for output in outputs])
        pred = sum([output[f'pred'] for output in outputs])
        true = sum([output[f'true'] for output in outputs])

        p = correct / pred if pred > 0 else 0
        r = correct / true if true > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r > 0) else 0

        return f1, {'p': p, 'r': r, 'f1': f1}


class Tree(Metric):
    def __init__(self, punct_idx):
        self.punct_idx = punct_idx

    def step(self, batch, preds, targets, mask) -> dict:
        parc, prel = preds
        rarc, rrel = targets

        parc[:, 0, 1:] = float('-inf')
        parc.diagonal(0, 1, 2).fill_(float('-inf'))
        parc = eisner(parc, torch.cat([torch.zeros_like(mask[:, :1], dtype=torch.bool), mask], dim=1))

        prel = torch.argmax(prel, dim=-1)
        prel = prel.gather(-1, parc.unsqueeze(-1)).squeeze(-1)

        # 对 punc 不计算分数
        punc_mask = rrel != self.punct_idx
        mask = mask & punc_mask

        arc_true = (parc == rarc)[mask]
        rel_true = (prel == rrel)[mask]
        union_true = arc_true & rel_true

        return {
            "arc_true": torch.sum(arc_true, dtype=torch.float).item(),
            "uni_true": torch.sum(union_true, dtype=torch.float).item(),
            "all": union_true.numel()
        }

    def epoch_end(self, outputs) -> Tuple[float, dict]:
        uas = sum([output["arc_true"] for output in outputs]) \
              / sum([output[f"all"] for output in outputs])
        las = sum([output[f"uni_true"] for output in outputs]) \
              / sum([output[f"all"] for output in outputs])

        return las, {'uas': uas, 'las': las}


class CharTree(Metric):
    # 这里假定了 app = 0
    def __init__(self, punct_idx):
        self.app_idx = 0
        self.punct_idx = punct_idx

    def get_graph_entities(self, batch_arcs, batch_labels, batch_lens):
        seg_entities = []
        arc_entities = []
        rel_entities = []
        for batch, (arcs, labels) in enumerate(zip(batch_arcs.cpu().numpy(), batch_labels.cpu().numpy())):
            sent_length = batch_lens[batch]

            start = 0
            re_heads = []
            re_labels = []

            words = []
            ranges = []
            for word_idx, (head, label) in enumerate(zip(arcs, labels)):
                if word_idx > sent_length - 1:
                    break
                ranges.append(len(words))
                if label != self.app_idx or word_idx == sent_length - 1:
                    words.append((batch, start, word_idx + 1))
                    re_heads.append(head)
                    re_labels.append(label)
                    start = word_idx + 1
                else:
                    continue

            seg_entities.extend(words)
            for word_idx, head in enumerate(re_heads):
                if re_labels[word_idx] == -1:
                    continue

                if head == 0:
                    arc_entities.append(('root', words[word_idx]))
                    rel_entities.append(('root', re_labels[word_idx], words[word_idx]))
                else:
                    head_word_idx = ranges[head - 1]
                    head_word_span = words[head_word_idx]

                    arc_entities.append((head_word_span, words[word_idx]))
                    rel_entities.append((head_word_span, re_labels[word_idx], words[word_idx]))

        return set(arc_entities), set(rel_entities), set(seg_entities)

    def step(self, batch, preds, targets, mask) -> dict:
        seq_lens = torch.sum(mask, dim=-1, dtype=torch.int).cpu().numpy()

        parc, prel = preds
        rarc, rrel = targets

        mask = torch.cat([torch.zeros_like(mask[:, :1], dtype=torch.bool), mask], dim=1)

        parc[:, 0, 1:] = float('-inf')
        parc.diagonal(0, 1, 2).fill_(float('-inf'))
        parc = eisner(parc, mask)

        # batch_size x max_len without root
        arange_index = torch.arange(1, mask.shape[1] + 1, dtype=torch.long, device=prel.device) \
            .unsqueeze(0) \
            .repeat(mask.shape[0], 1)
        app_masks = parc.ne(arange_index)
        app_masks = app_masks.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, prel.shape[-1])
        app_masks[:, :, :, 1:] = 0

        prel = prel.masked_fill(app_masks, float("-inf"))
        prel = torch.argmax(prel, dim=-1)
        prel = prel.gather(-1, parc.unsqueeze(-1)).squeeze(-1)

        parc = parc[:, 1:]
        prel = prel[:, 1:]

        punc_mask = rrel == self.punct_idx
        rrel = rrel.masked_fill(punc_mask, -1)
        prel = prel.masked_fill(punc_mask, -1)

        rarc_entities, rrel_entities, rseg_entities = self.get_graph_entities(rarc, rrel, seq_lens)
        parc_entities, prel_entities, pseg_entities = self.get_graph_entities(parc, prel, seq_lens)

        return {
            'uas_correct': len(parc_entities & rarc_entities),
            'uas_pred': len(parc_entities),
            'uas_true': len(rarc_entities),
            'las_correct': len(prel_entities & rrel_entities),
            'las_pred': len(prel_entities),
            'las_true': len(rrel_entities),
            'seg_correct': len(pseg_entities & rseg_entities),
            'seg_pred': len(pseg_entities),
            'seg_true': len(rseg_entities),
        }

    def epoch_end(self, outputs) -> Tuple[float, dict]:
        uas_f1, uas_metric = self.preffix_compute('uas', outputs)
        las_f1, las_metric = self.preffix_compute('las', outputs)
        seg_f1, seg_metric = self.preffix_compute('seg', outputs)

        las_metric.update(uas_metric)
        las_metric.update(seg_metric)
        return las_f1, las_metric

    def preffix_compute(self, preffix, outputs):
        correct = sum([output[f'{preffix}_correct'] for output in outputs])
        pred = sum([output[f'{preffix}_pred'] for output in outputs])
        true = sum([output[f'{preffix}_true'] for output in outputs])

        p = correct / pred if pred > 0 else 0
        r = correct / true if true > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r > 0) else 0

        return f1, {f'{preffix}_p': p, f'{preffix}_r': r, f'{preffix}_f1': f1}
