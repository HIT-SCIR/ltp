import torch
from torch import nn
from transformers.models.roformer.modeling_roformer import (
    RoFormerSelfAttention,
    RoFormerSinusoidalPositionalEmbedding,
)

INF = 1e12


class GlobalPointer(nn.Module):
    """全局指针模块 将序列的每个(start, end)作为整体来进行判断 参考：https://kexue.fm/archives/8373."""

    def __init__(
        self,
        input_size,
        num_labels,
        head_size=64,
        RoPE=True,
        tril_mask=False,
        max_length=512,
    ):
        super().__init__()
        self.heads = num_labels
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(input_size, num_labels * 2 * head_size)
        if RoPE:
            self.rotary = RoFormerSinusoidalPositionalEmbedding(max_length, head_size)

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        inputs = inputs.view(bs, seqlen, self.heads, 2, self.head_size)
        qw, kw = inputs.unbind(axis=-2)

        # method 1
        # inputs = inputs.reshape(bs, seqlen, self.num_labels, 2, self.head_size)
        # qw, kw = inputs.unbind(axis=-2)

        # method 2
        # inputs = inputs.reshape(bs, seqlen, self.num_labels, 2 * self.head_size)
        # qw, kw = inputs.chunk(2, axis=-1)

        # original
        # inputs = inputs.chunk(self.num_labels, axis=-1)
        # inputs = torch.stack(inputs, axis=-2)
        # qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            sinusoidal_pos = self.rotary(inputs.shape[:2])[None, :, None, :]
            qw, kw = RoFormerSelfAttention.apply_rotary_position_embeddings(sinusoidal_pos, qw, kw)

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = 1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            logits = logits - attn_mask * INF

        # 排除下三角
        if self.tril_mask:
            # method 1
            # logits.tril(diagonal=-1).sub_(INF)

            # original
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * INF

        # scale返回
        return logits / self.head_size**0.5


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer 参考：https://kexue.fm/archives/8877."""

    def __init__(
        self,
        input_size,
        num_labels,
        head_size=64,
        RoPE=True,
        tril_mask=False,
        max_length=512,
    ):
        super().__init__()
        self.heads = num_labels
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(input_size, head_size * 2)
        self.dense2 = nn.Linear(head_size * 2, num_labels * 2)
        if RoPE:
            self.rotary = RoFormerSinusoidalPositionalEmbedding(max_length, head_size)

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            sinusoidal_pos = self.rotary(inputs.shape[:2])[None, :, :]
            qw, kw = RoFormerSelfAttention.apply_rotary_position_embeddings(sinusoidal_pos, qw, kw)

        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size**0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  # 'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = 1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            logits = logits - attn_mask * INF

        # 排除下三角
        if self.tril_mask:
            # method 1
            # logits.tril(diagonal=-1).sub_(INF)

            # original
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * INF

        return logits


def main():
    gp = GlobalPointer(
        input_size=128,
        num_labels=12,
        head_size=64,
        RoPE=True,
        tril_mask=True,
        max_length=512,
    )
    egp = EfficientGlobalPointer(
        input_size=128,
        num_labels=12,
        head_size=64,
        RoPE=True,
        tril_mask=True,
        max_length=512,
    )
    inputs = torch.randn(2, 512, 128)
    mask = torch.randint(0, 2, (2, 512)).sort(dim=-1, descending=True)[0]
    gp_outputs = gp(inputs, mask)
    egp_outputs = egp(inputs, mask)
    print(gp_outputs.shape)
    print(egp_outputs.shape)


if __name__ == "__main__":
    main()
