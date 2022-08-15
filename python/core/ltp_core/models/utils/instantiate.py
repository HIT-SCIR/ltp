import functools
import importlib
from typing import Callable


def find_callable(target: str) -> Callable:
    target_module_path, target_callable_path = target.rsplit(".", 1)
    target_callable_paths = [target_callable_path]

    target_module = None
    while len(target_module_path):
        try:
            target_module = importlib.import_module(target_module_path)
            break
        except Exception as e:
            target_module_path, target_callable_path = target_module_path.rsplit(".", 1)
            if len(target_module_path) == 0:
                raise e
            target_callable_paths.append(target_callable_path)
    target_callable = target_module
    for attr in reversed(target_callable_paths):
        target_callable = getattr(target_callable, attr)

    return target_callable


def instantiate(config, target="_ltp_target_", partial="_ltp_partial_"):
    if isinstance(config, dict) and target in config:
        target_path = config.get(target)
        target_callable = find_callable(target_path)

        is_partial = config.get(partial, False)
        target_args = {
            key: instantiate(value)
            for key, value in config.items()
            if key not in [target, partial]
        }

        if is_partial:
            return functools.partial(target_callable, **target_args)
        else:
            return target_callable(**target_args)
    elif isinstance(config, dict):
        return {key: instantiate(value) for key, value in config.items()}
    else:
        return config


def instantiate_omega(config, target="_ltp_target_", partial="_ltp_partial_"):
    from omegaconf import DictConfig

    if (isinstance(config, dict) or isinstance(config, DictConfig)) and target in config:
        target_path = config.get(target)
        target_callable = find_callable(target_path)

        is_partial = config.get(partial, False)
        target_args = {
            key: instantiate_omega(value)
            for key, value in config.items()
            if key not in [target, partial]
        }

        if is_partial:
            return functools.partial(target_callable, **target_args)
        else:
            return target_callable(**target_args)
    elif isinstance(config, dict) or isinstance(config, DictConfig):
        return {key: instantiate_omega(value) for key, value in config.items()}
    else:
        return config


def main():
    import yaml

    with open("configs/model/model.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
            model_config = config["model"]
        except yaml.YAMLError as exc:
            print(exc)

    model = instantiate(model_config)
    print(model)


if __name__ == "__main__":
    main()
