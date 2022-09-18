from collections import namedtuple

from torch import nn

from ltp_core.models.nn.mlp import MLP

SentClassifierResult = namedtuple("SentClassifierResult", ["logits"])


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_labels,
        dropout=0.1,
        hidden_sizes=None,
    ):
        super().__init__()
        if hidden_sizes is not None:
            self.classifier = MLP([input_size, *hidden_sizes, num_labels], dropout=dropout)
        else:
            self.classifier = MLP([input_size, num_labels], dropout=dropout)

    def forward(self, hidden_states, attention_mask=None) -> SentClassifierResult:
        logits = self.classifier(hidden_states)
        return SentClassifierResult(logits=logits)
