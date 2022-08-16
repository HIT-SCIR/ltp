from collections import namedtuple
from typing import Optional

from torch import nn

from ltp_core.models.nn.biaffine import Biaffine
from ltp_core.models.nn.crf import CRF
from ltp_core.models.nn.mlp import MLP
from ltp_core.models.nn.relative_transformer import RelativeTransformer

TokenClassifierResult = namedtuple("TokenClassifierResult", ["logits", "attention_mask", "crf"])


class MLPTokenClassifier(nn.Module):
    crf: Optional[CRF]

    def __init__(
        self,
        input_size,
        num_labels,
        dropout=0.1,
        hidden_sizes=None,
        use_crf=False,
        crf_reduction="sum",
    ):
        super().__init__()
        if hidden_sizes is not None:
            self.classifier = MLP([input_size, *hidden_sizes, num_labels], dropout=dropout)
        else:
            self.classifier = MLP([input_size, num_labels], dropout=dropout)
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True, reduction=crf_reduction)
        else:
            self.crf = None

    def forward(self, hidden_states, attention_mask=None) -> TokenClassifierResult:
        logits = self.classifier(hidden_states)
        return TokenClassifierResult(logits=logits, attention_mask=attention_mask, crf=self.crf)


class RelTransformerTokenClassifier(nn.Module):
    crf: Optional[CRF]

    def __init__(
        self,
        input_size,
        num_labels,
        hidden_size=256,
        dropout=0.1,
        num_heads=4,
        num_layers=2,
        max_length=512,
        use_crf=False,
        crf_reduction="sum",
    ):
        super().__init__()
        self.relative_transformer = RelativeTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_length * 2,
        )
        self.classifier = MLP([input_size, num_labels], dropout=dropout)
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True, reduction=crf_reduction)
        else:
            self.crf = None

    def forward(self, hidden_states, attention_mask=None) -> TokenClassifierResult:
        logits = self.relative_transformer(hidden_states, attention_mask)
        logits = self.classifier(logits)
        return TokenClassifierResult(logits=logits, attention_mask=attention_mask, crf=self.crf)


class BiaffineTokenClassifier(nn.Module):
    crf: Optional[CRF]

    def __init__(
        self,
        input_size,
        num_labels,
        dropout=0.1,
        hidden_size=None,
        hidden_sizes=None,
        use_crf=False,
        crf_reduction="sum",
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size
        if hidden_sizes is not None:
            layer_sizes = [input_size, *hidden_sizes, hidden_size * 2]
        else:
            layer_sizes = [input_size, hidden_size * 2]
        self.mlp = MLP(layer_sizes, output_dropout=dropout, output_activation=nn.ReLU)
        self.atten = Biaffine(hidden_size, hidden_size, num_labels, bias_x=True, bias_y=True)
        self.hidden_size = hidden_size

        if use_crf:
            self.crf = CRF(num_labels, batch_first=True, reduction=crf_reduction)
        else:
            self.crf = None

    def forward(self, hidden_states, attention_mask=None) -> TokenClassifierResult:
        bs, seqlen = hidden_states.shape[:2]
        logits = self.mlp(hidden_states)
        logits = logits.view(bs, seqlen, 2, self.hidden_size)
        logits_h, logits_d = logits.unbind(axis=-2)
        logits = self.atten(logits_h, logits_d).permute(0, 2, 3, 1)

        return TokenClassifierResult(logits=logits, attention_mask=attention_mask, crf=self.crf)
