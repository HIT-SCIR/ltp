from argparse import ArgumentParser
from pathlib import Path
from os import path

import pickle
from types import ModuleType

import torch


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "pytorch_lightning.utilities.argparse_utils":
            renamed_module = "pytorch_lightning.utilities.argparse"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


class RenamePickleModule(ModuleType):
    def __init__(self):
        super().__init__('pickle')

    def __getattr__(self, item):
        if item == 'Unpickler':
            return RenameUnpickler

        return getattr(pickle, item)


def patch(ckpt_path, patched_path=None):
    if patched_path is None:
        ckpt_path_utils = Path(ckpt_path)
        patched_path = path.join(ckpt_path_utils.parent, f"patched_{ckpt_path_utils.name}")

    if path.exists(patched_path):
        return patched_path
    ckpt = torch.load(ckpt_path, map_location='cpu', pickle_module=RenamePickleModule())
    torch.save(ckpt, patched_path)
    return patched_path


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--patched", type=str, default=None)
    args = parser.parse_args()

    patch(args.ckpt, args.patched)


if __name__ == '__main__':
    main()
