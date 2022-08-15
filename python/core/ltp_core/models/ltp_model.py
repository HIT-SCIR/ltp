from typing import Dict

import torch
from torch import nn
from torch.nn import ModuleDict
from transformers.modeling_outputs import BaseModelOutput


class LTPModule(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        heads: Dict[str, nn.Module],
        processor: Dict[str, nn.Module],
    ):
        super().__init__()
        self.backbone = backbone
        self.processor = ModuleDict(processor)
        self.task_heads = ModuleDict(heads)

    def forward(
        self,
        task_name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        word_index: torch.Tensor = None,
        word_attention_mask: torch.Tensor = None,
    ):
        outputs: BaseModelOutput = self.backbone(input_ids, attention_mask, token_type_ids)
        hidden_state, attention_mask = self.processor[task_name](
            outputs, attention_mask, word_index, word_attention_mask
        )
        return self.task_heads[task_name](hidden_state, attention_mask)
