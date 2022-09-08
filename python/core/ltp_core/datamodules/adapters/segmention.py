from ltp_core.datamodules.components.conllu import Conllu
from ltp_core.datamodules.utils.datasets import load_dataset

PREFIX_B = 0
PREFIX_I = 1
PREFIX_M = 1
PREFIX_E = 2
PREFIX_S = 3


def length2bi(length):
    if length == 0:
        return []
    elif length == 1:
        return [PREFIX_B]
    elif length == 2:
        return [PREFIX_B, PREFIX_I]
    else:
        return [PREFIX_B] + [PREFIX_I] * (length - 1)


def length2bmes(length):
    if length == 0:
        return []
    elif length == 1:
        return [PREFIX_S]
    elif length == 2:
        return [PREFIX_B, PREFIX_E]
    elif length == 3:
        return [PREFIX_B, PREFIX_M, PREFIX_E]
    else:
        return [PREFIX_B] + [PREFIX_M] * (length - 2) + [PREFIX_E]


def tokenize(examples, tokenizer, max_length, length2labels=length2bi):
    res = tokenizer(
        examples["form"],
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
    )
    labels = []
    for encoding in res.encodings:
        labels.append([])
        last_word_idx = -1
        word_length = 0
        for word_idx in encoding.words[1:-1]:
            if word_idx == last_word_idx:
                word_length += 1
            else:
                labels[-1].extend(length2labels(word_length))
                last_word_idx = word_idx
                word_length = 1
        labels[-1].extend(length2labels(word_length))

    result = res.data
    for ids in res["input_ids"]:
        ids[0] = tokenizer.cls_token_id
        ids[-1] = tokenizer.sep_token_id
    result["overflow"] = [len(encoding.overflowing) > 0 for encoding in res.encodings]
    result["labels"] = labels
    return result


def build_dataset(data_dir, task_name, tokenizer, max_length=512, mode="bmes", **kwargs):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dataset = load_dataset(Conllu, data_dir=data_dir, cache_dir=data_dir)
    dataset = dataset.remove_columns(
        ["id", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
    )
    if mode == "bmes":
        dataset = dataset.map(
            lambda examples: tokenize(examples, tokenizer, max_length, length2bmes),
            batched=True,
        )
    elif mode == "bi":
        dataset = dataset.map(
            lambda examples: tokenize(examples, tokenizer, max_length, length2bi),
            batched=True,
        )
    else:
        raise NotImplementedError(f"not supported {mode} mode")
    dataset = dataset.filter(lambda x: not x["overflow"])
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    return dataset


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
    dataset = build_dataset(data_dir="data/seg", task_name="seg", tokenizer=tokenizer, mode="bmes")
    print(dataset)


if __name__ == "__main__":
    main()
