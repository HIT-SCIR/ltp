from ltp_core.datamodules.adapters.postagger import tokenize
from ltp_core.datamodules.components.bio import Bio
from ltp_core.datamodules.utils.datasets import load_dataset


def build_dataset(data_dir, task_name, tokenizer, max_length=512, **kwargs):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dataset = load_dataset(Bio, data_dir=data_dir, cache_dir=data_dir)
    dataset = dataset.rename_column("bio", "labels")
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
