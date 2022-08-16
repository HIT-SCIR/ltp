#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import torch
from torch._six import string_classes
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)

_TORCH_MAJOR, _TORCH_MINOR = map(int, torch.__version__.split(".")[0:2])

if _TORCH_MAJOR < 1 or (_TORCH_MAJOR == 1 and _TORCH_MINOR < 8):
    from torch._six import container_abcs, int_classes
else:
    int_classes = int
    import collections.abc as container_abcs


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        try:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel, device=elem.device)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
        except Exception as e:
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        batch = [torch.stack(it) for it in batch]
        elem_sizes = [it.shape for it in batch]
        max_sizes = (max(sizes) for sizes in zip(*elem_sizes))
        batched = torch.zeros(len(batch), *max_sizes, dtype=batch[0].dtype)
        for idx, (elem, elem_size) in enumerate(zip(batch, elem_sizes)):
            size_1, size_2 = elem_size
            batched[idx, :size_1, :size_2] = elem
        return batched

    raise TypeError(default_collate_err_msg_format.format(elem_type))
