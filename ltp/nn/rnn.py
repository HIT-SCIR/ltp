#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    __RNN__ = nn.RNN

    def __init__(self, input_size, hidden_size, num_layers=1,
                 bias=True, batch_first=True, dropout=0., bidirectional=True, **kwargs):
        super(RNN, self).__init__()
        self.rnn = self.__RNN__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            **kwargs
        )

    @property
    def batch_first(self):
        return self.rnn.batch_first

    @property
    def batch_dim(self):
        return 0 if self.batch_first else 1

    @property
    def seq_dim(self):
        return 1 if self.batch_first else 0

    @property
    def num_layers(self):
        return self.rnn.num_layers

    @property
    def bidirectional(self):
        return self.rnn.bidirectional

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    @property
    def output_size(self):
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size

    @property
    def num_directions(self):
        return 2 if self.bidirectional else 1

    def forward(self, inputs: Tensor, hx=None, seq_lens: Tensor = None, process=False):
        if seq_lens is None:
            batch_num, seq_length = inputs.shape[self.batch_dim], inputs.shape[self.seq_dim]
            seq_lens = torch.full((batch_num,), seq_length, device=inputs.device)

        # sort
        sort_lens, sort_idx = torch.sort(seq_lens, dim=0, descending=True)
        inputs = torch.index_select(inputs, dim=self.batch_dim, index=sort_idx)

        # pack
        packed_inputs = pack_padded_sequence(inputs, sort_lens, batch_first=self.batch_first)
        packed_outputs, hidden_last = self.rnn(packed_inputs, hx=hx)  # -> [N,L,C]

        # unpack
        outputs, lengths = pad_packed_sequence(packed_outputs, batch_first=self.batch_first)

        # unsort
        _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)

        def unsort(inputs, dim=self.batch_dim):
            return torch.index_select(inputs, dim=dim, index=unsort_idx)

        if process:
            def process_fn(hidden):
                hidden = hidden.contiguous().view(
                    self.num_layers, self.num_directions,
                    inputs.shape[self.batch_dim], self.hidden_size)
                hidden = hidden[-1]  # (direction, batch, hidden/2) choose last layer
                if self.bidirectional:  # concat direction dim
                    return torch.cat((hidden[0], hidden[1]), -1)
                else:
                    return hidden[0]
        else:
            process_fn = lambda x: x

        if not isinstance(hidden_last, Tensor):
            hidden_last, cell_last = hidden_last
            return unsort(outputs), (unsort(process_fn(hidden_last), 1),
                                     unsort(process_fn(cell_last), 1))
        else:
            return unsort(outputs), unsort(process_fn(hidden_last), 1)


class LSTM(RNN):
    __RNN__ = nn.LSTM


class GRU(RNN):
    __RNN__ = nn.GRU
