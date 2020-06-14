#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
from typing import Optional
import torch, math
from torch import Tensor, nn
import torch.nn.functional as F
from ltp.utils import length_to_mask
from ltp.nn import ffnn


class RelativeEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        """
        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.as_tensor(1.0))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings // 2, num_embeddings // 2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings // 2 + 1
        return emb

    def forward(self, inputs: Tensor):
        """Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = inputs.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(inputs.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, input_size, num_head, dropout, r_w_bias=None, r_r_bias=None, max_length=1024):
        """
        :param int input_size:
        :param int num_head:
        :param dropout: 对attention map的dropout
        :param r_w_bias: n_head x head_dim or None
        :param r_r_bias: n_head x head_dim or None
        """
        super().__init__()
        self.qv_linear = nn.Linear(input_size, input_size * 2, bias=False)
        self.n_head = num_head
        self.head_dim = input_size // num_head
        self.dropout_layer = nn.Dropout(dropout)

        self.pos_embed = RelativeEmbedding(input_size // num_head, 0, max_length)

        self.scale = 1

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(num_head, input_size // num_head)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(num_head, input_size // num_head)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x d_model
        :param mask: batch_size x max_len
        :return:
        """

        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)  # l x head_dim

        qv = self.qv_linear(x)  # batch_size x max_len x d_model2
        q, v = torch.chunk(qv, chunks=2, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = x.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', rw_head_q, k)  # b x n x l x d, n是head

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BD = self._shift(BD)
        attn = AC + BD

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)  # b x n x l x d

        return v

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        转换为
        0   1  2
        -1  0  1
        -2 -1  0
        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD


class RelativeTransformerLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout=0.2, after_norm=True, max_length=1024):
        """
        :param int input_size: 一般512之类的
        :param int hidden_size: FFN中间层的dimension的大小\
        :param num_heads: self attention模块
        :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.after_norm = after_norm
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.self_attn = RelativeMultiHeadAttn(input_size, num_heads, dropout=dropout, max_length=max_length)
        self.ffn = ffnn(input_size, input_size, hidden_size, dropout, depth=2, last_dropout=True)

    def forward(self, x, mask):
        """
        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class RelativeTransformer(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, num_heads, dropout, after_norm=True, max_length=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTransformerLayer(input_size, hidden_size, num_heads, dropout, after_norm, max_length=max_length)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, length: Tensor, gold: Optional = None):
        """
        :param x: batch_size x max_len
        :param length: sequence length, B
        """

        mask = length_to_mask(length, dtype=torch.long)

        for layer in self.layers:
            x = layer(x, mask)
        return x, length, gold
