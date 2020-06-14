from . import Sequence
from ltp.utils import length_to_mask


class Segment(Sequence):
    def __init__(self, *args):
        super(Segment, self).__init__(['B-W', 'I-W'])

    def step(self, y_pred, y: dict):
        mask = ~ length_to_mask(y['text_length'])
        target = y['word_idn']
        target[mask] = -1
        super(Segment, self).step(y_pred, target)

    @classmethod
    def from_extra(cls, extra: dict):
        return {}
