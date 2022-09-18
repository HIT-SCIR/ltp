import os.path

from huggingface_hub import CommitOperationAdd, HfApi


def upload_model(model_dir, repo_id):
    api = HfApi()
    operations = [
        CommitOperationAdd(
            path_in_repo="config.json",
            path_or_fileobj=os.path.join(model_dir, "config.json"),
        ),
        CommitOperationAdd(
            path_in_repo="pytorch_model.bin",
            path_or_fileobj=os.path.join(model_dir, "pytorch_model.bin"),
        ),
        CommitOperationAdd(
            path_in_repo="vocab.txt",
            path_or_fileobj=os.path.join(model_dir, "vocab.txt"),
        ),
    ]

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Uploaded model",
    )


def upload_tokenizer(model_dir, repo_id):
    api = HfApi()
    operations = [
        CommitOperationAdd(
            path_in_repo="added_tokens.json",
            path_or_fileobj=os.path.join(model_dir, "added_tokens.json"),
        ),
        CommitOperationAdd(
            path_in_repo="special_tokens_map.json",
            path_or_fileobj=os.path.join(model_dir, "special_tokens_map.json"),
        ),
        CommitOperationAdd(
            path_in_repo="tokenizer.json",
            path_or_fileobj=os.path.join(model_dir, "tokenizer.json"),
        ),
        CommitOperationAdd(
            path_in_repo="tokenizer_config.json",
            path_or_fileobj=os.path.join(model_dir, "tokenizer_config.json"),
        ),
    ]

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Uploaded tokenizer",
    )


def upload_readme(model_dir, repo_id):
    api = HfApi()
    operations = [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj="README.md",
        ),
    ]

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Uploaded model",
    )


def main():
    for model in ["legacy", "tiny", "small", "base", "base1", "base2"]:
        upload_readme(None, repo_id=f"LTP/{model}")


if __name__ == "__main__":
    main()
