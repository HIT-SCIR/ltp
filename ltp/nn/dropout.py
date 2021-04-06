# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    r"""
    SharedDropout differs from the vanilla dropout strategy in that
    the dropout mask is shared across one dimension.
    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.
        batch_first (bool):
            If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
            Default: ``True``.
    Examples:
        >>> x = torch.ones(1, 3, 5)
        >>> nn.Dropout()(x)
        tensor([[[0., 2., 2., 0., 0.],
                 [2., 2., 0., 2., 2.],
                 [2., 2., 2., 2., 0.]]])
        >>> SharedDropout()(x)
        tensor([[[2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.]]])
    """

    def __init__(self, p=0.5, batch_first=True):
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                A tensor of any shape.
        Returns:
            The returned tensor is of the same shape as `x`.
        """

        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
            else:
                mask = self.get_mask(x[0], self.p)
            x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)


class IndependentDropout(nn.Module):
    r"""
    For :math:`N` tensors, they use different dropout masks respectively.
    When :math:`N-M` of them are dropped, the remaining :math:`M` ones are scaled by a factor of :math:`N/M` to compensate,
    and when all of them are dropped together, zeros are returned.
    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.
    Examples:
        >>> x, y = torch.ones(1, 3, 5), torch.ones(1, 3, 5)
        >>> x, y = IndependentDropout()(x, y)
        >>> x
        tensor([[[1., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 0.],
                 [2., 2., 2., 2., 2.]]])
        >>> y
        tensor([[[1., 1., 1., 1., 1.],
                 [2., 2., 2., 2., 2.],
                 [0., 0., 0., 0., 0.]]])
    """

    def __init__(self, p=0.5):
        super().__init__()

        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, *items):
        r"""
        Args:
            items (list[~torch.Tensor]):
                A list of tensors that have the same shape except the last dimension.
        Returns:
            The returned tensors are of the same shape as `items`.
        """

        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(-1) for item, mask in zip(items, masks)]

        return items