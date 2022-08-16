from ltp_core.datamodules.components.conllu import Conllu
from ltp_core.datamodules.utils.datasets import load_dataset


def tokenize(examples, tokenizer, max_length, char_base=False):
    """

    Args:
        examples:
        tokenizer:
        max_length:
        char_base: 这里指的是 examples 中的 form[即 word]是否是字级别的

    Returns:

    """
    res = tokenizer.batch_encode_plus(
        examples["form"],
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
    )
    result = res.data
    for ids in result["input_ids"]:
        ids[0] = tokenizer.cls_token_id
        ids[-1] = tokenizer.sep_token_id
    result["overflow"] = [len(encoding.overflowing) > 0 for encoding in res.encodings]

    if not char_base:
        word_index = []
        for encoding in res.encodings:
            word_index.append([])

            last_word_idx = -1
            current_length = 0
            for word_idx in encoding.words[1:-1]:
                if word_idx != last_word_idx:
                    word_index[-1].append(current_length)
                current_length += 1
                last_word_idx = word_idx
        result["word_index"] = word_index
        result["word_attention_mask"] = [[True] * len(index) for index in word_index]
    return result


def build_dataset(data_dir, task_name, tokenizer, max_length=512, **kwargs):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dataset = load_dataset(Conllu, data_dir=data_dir, cache_dir=data_dir)
    dataset = dataset.remove_columns(
        ["id", "lemma", "upos", "feats", "head", "deprel", "deps", "misc"]
    )
    dataset = dataset.rename_column("xpos", "labels")
    dataset = dataset.map(lambda examples: tokenize(examples, tokenizer, max_length), batched=True)
    dataset = dataset.filter(lambda x: not x["overflow"])
    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "word_index",
            "word_attention_mask",
            "labels",
        ],
    )
    return dataset
