from ltp.utils import get_entities


def test_span():
    span = ['B-PER', 'I-PER', 'O', 'B-LOC']
    assert get_entities(span) == [('PER', 0, 1), ('LOC', 3, 3)]
