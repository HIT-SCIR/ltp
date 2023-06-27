from datasets import load_dataset


# todo: implement
def build_dataset(task_name):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    load_dataset("glue", task_name)
