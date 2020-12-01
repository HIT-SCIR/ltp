import torch
from torch import Tensor


def length_to_mask(length: Tensor, max_len: int = None, dtype=None):
    """
    将 Sequence length 转换成 Mask

    >>> lens = [3, 5, 4]
    >>> length_to_mask(length)
    >>> [[1, 1, 1, 0, 0],\
        [1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 0]]

    :param length: [batch,]
    :param max_len: 最大长度
    :param dtype: nn.dtype
    :return: batch * max_len : 如果 max_len is None
    :return: batch * max(length) : 如果 max_len is None
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len is None:
        max_len = max_len or torch.max(length)
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype). \
               expand(length.shape[0], max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask
