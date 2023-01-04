# ! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import warnings


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ["O", "B", "I", "M", "E", "S"]:
            return

        if suffix:
            if not chunk.endswith(("-B", "-I", "-M", "-E", "-S")):
                warnings.warn(f"{chunk} seems not to be NE tag.")

        else:
            if not chunk.startswith(("B-", "I-", "M-", "E-", "S-")):
                warnings.warn(f"{chunk} seems not to be NE tag.")

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ["O"]]

    prev_tag = "O"
    prev_type = ""
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit("-", maxsplit=1)[0] or "_"
        else:
            tag = chunk[0]
            type_ = chunk[1:].split("-", maxsplit=1)[-1] or "_"

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == "E":
        chunk_end = True
    if prev_tag == "S":
        chunk_end = True

    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "S":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if (prev_tag == "I" or prev_tag == "M") and tag == "B":
        chunk_end = True
    if (prev_tag == "I" or prev_tag == "M") and tag == "S":
        chunk_end = True
    if (prev_tag == "I" or prev_tag == "M") and tag == "O":
        chunk_end = True

    if prev_tag != "O" and prev_tag != "." and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == "B":
        chunk_start = True
    if tag == "S":
        chunk_start = True

    if prev_tag == "E" and tag == "E":
        chunk_start = True
    if prev_tag == "E" and (prev_tag == "I" or prev_tag == "M"):
        chunk_start = True
    if prev_tag == "S" and tag == "E":
        chunk_start = True
    if prev_tag == "S" and (prev_tag == "I" or prev_tag == "M"):
        chunk_start = True
    if prev_tag == "O" and tag == "E":
        chunk_start = True
    if prev_tag == "O" and (prev_tag == "I" or prev_tag == "M"):
        chunk_start = True

    if tag != "O" and tag != "." and prev_type != type_:
        chunk_start = True

    return chunk_start
