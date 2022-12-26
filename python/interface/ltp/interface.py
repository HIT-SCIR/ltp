import json
import os
from typing import Dict, Optional, Union

from ltp.legacy import LTP as LTP_legacy
from ltp.mixin import CONFIG_NAME
from ltp.nerual import LTP as LTP_neural
from ltp.utils import get_pylogger

logger = get_pylogger()


def LTP(
    pretrained_model_name_or_path="LTP/small",
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Dict = None,
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    **model_kwargs,
) -> Union[LTP_legacy, LTP_neural]:
    r"""
    Instantiate a pretrained LTP model from a pre-trained model
            configuration from huggingface-hub. The model is set in
            evaluation mode by default using `model.eval()` (Dropout modules
            are deactivated). To train the model, you should first set it
            back in training mode with `model.train()`.

            Parameters:
                pretrained_model_name_or_path (`str` or `os.PathLike`):
                    Can be either:
                        - A string, the `model id` of a pretrained model
                          hosted inside a model repo on huggingface.co.
                          Valid model ids are [`LTP/tiny`, `LTP/small`,
                          `LTP/base`, `LTP/base1`, `LTP/base1`, `LTP/legacy`
                          ], the legacy model only support cws, pos and ner,
                          but more fast.
                        - You can add `revision` by appending `@` at the end
                          of model_id simply like this:
                          `dbmdz/bert-base-german-cased@main` Revision is
                          the specific model version to use. It can be a
                          branch name, a tag name, or a commit id, since we
                          use a git-based system for storing models and
                          other artifacts on huggingface.co, so `revision`
                          can be any identifier allowed by git.
                        - A path to a `directory` containing model weights
                          saved using
                          [`~transformers.PreTrainedModel.save_pretrained`],
                          e.g., `./my_model_directory/`.
                        - `None` if you are both providing the configuration
                          and state dictionary (resp. with keyword arguments
                          `config` and `state_dict`).
                force_download (`bool`, *optional*, defaults to `False`):
                    Whether to force the (re-)download of the model weights
                    and configuration files, overriding the cached versions
                    if they exist.
                resume_download (`bool`, *optional*, defaults to `False`):
                    Whether to delete incompletely received files. Will
                    attempt to resume the download if such a file exists.
                proxies (`Dict[str, str]`, *optional*):
                    A dictionary of proxy servers to use by protocol or
                    endpoint, e.g., `{'http': 'foo.bar:3128',
                    'http://hostname': 'foo.bar:4012'}`. The proxies are
                    used on each request.
                use_auth_token (`str` or `bool`, *optional*):
                    The token to use as HTTP bearer authorization for remote
                    files. If `True`, will use the token generated when
                    running `transformers-cli login` (stored in
                    `~/.huggingface`).
                cache_dir (`Union[str, os.PathLike]`, *optional*):
                    Path to a directory in which a downloaded pretrained
                    model configuration should be cached if the standard
                    cache should not be used.
                local_files_only(`bool`, *optional*, defaults to `False`):
                    Whether to only look at local files (i.e., do not try to
                    download the model).
                model_kwargs (`Dict`, *optional*):
                    model_kwargs will be passed to the model during
                    initialization

            <Tip>

            Passing `use_auth_token=True` is required when you want to use a
            private model.

            </Tip>
    """
    model_id = pretrained_model_name_or_path

    revision = None
    if len(model_id.split("@")) == 2:
        model_id, revision = model_id.split("@")

    if os.path.isdir(model_id) and CONFIG_NAME in os.listdir(model_id):
        config_file = os.path.join(model_id, CONFIG_NAME)
    else:
        from huggingface_hub.file_download import hf_hub_download
        from requests import RequestException

        try:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename=CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )
        except RequestException:
            logger.warning(f"{CONFIG_NAME} not found in HuggingFace Hub")
            config_file = None

    if config_file is not None:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
        model_kwargs.update({"config": config})
    else:
        raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

    if config["nerual"]:
        return LTP_neural._from_pretrained(
            model_id,
            revision,
            cache_dir,
            force_download,
            proxies,
            resume_download,
            local_files_only,
            use_auth_token,
            **model_kwargs,
        )
    else:
        return LTP_legacy._from_pretrained(
            model_id,
            revision,
            cache_dir,
            force_download,
            proxies,
            resume_download,
            local_files_only,
            use_auth_token,
            **model_kwargs,
        )
