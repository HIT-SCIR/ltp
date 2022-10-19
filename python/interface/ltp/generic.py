#! /usr/bin/env python
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, List, Optional, Tuple, Union


class ModelOutput(OrderedDict):
    """Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by
    integer or slice (like a tuple) or strings (like a dictionary) that will ignore the `None`
    attributes. Otherwise behaves like a regular python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """Convert self to a tuple containing all the attributes/keys that are not `None`."""
        return tuple(self[k] for k in self.keys())


@dataclass
class LTPOutput(ModelOutput):
    cws: Optional[Union[List[str], List[List[str]]]] = None
    pos: Optional[Union[List[str], List[List[str]]]] = None
    ner: Optional[Union[List[str], List[List[str]]]] = None
    srl: Optional[Union[List[str], List[List[str]]]] = None
    dep: Optional[Union[List[str], List[List[str]]]] = None
    sdp: Optional[Union[List[str], List[List[str]]]] = None
    sdpg: Optional[Union[List[str], List[List[str]]]] = None
