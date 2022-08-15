import torch
from torch import nn


class NOP(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, outputs, attention_mask=None, word_index=None, word_attention_mask=None):
        return self.dropout(outputs.last_hidden_state), attention_mask == 1


class TokenOnly(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, outputs, attention_mask=None, word_index=None, word_attention_mask=None):
        return (
            self.dropout(outputs.last_hidden_state[:, 1:-1]),
            attention_mask[:, 2:] == 1,
        )


class WordsOnly(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, outputs, attention_mask=None, word_index=None, word_attention_mask=None):
        hidden = outputs.last_hidden_state
        hidden = torch.gather(
            hidden[:, 1:-1, :],
            dim=1,
            index=word_index.unsqueeze(-1).expand(-1, -1, hidden.size(-1)),
        )
        return self.dropout(hidden), word_attention_mask


class ClsOnly(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, outputs, attention_mask=None, word_index=None, word_attention_mask=None):
        return self.dropout(outputs.last_hidden_state[:, 0]), None


class WordsWithHead(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, outputs, attention_mask=None, word_index=None, word_attention_mask=None):
        hidden = outputs.last_hidden_state
        hidden = torch.cat(
            [
                hidden[:, :1, :],
                torch.gather(
                    hidden[:, 1:-1, :],
                    dim=1,
                    index=word_index.unsqueeze(-1).expand(-1, -1, hidden.size(-1)),
                ),
            ],
            dim=1,
        )
        return self.dropout(hidden), word_attention_mask
