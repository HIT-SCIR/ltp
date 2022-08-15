from typing import Optional, Union

import torch
from torch.nn import Module


class BaseModule(Module):
    __jit_unused_properties__ = ["device", "dtype"]

    def __init__(self):
        super().__init__()
        self._dtype = torch.get_default_dtype()
        self._device = torch.device("cpu")

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]):
        # necessary to avoid infinite recursion
        raise RuntimeError("Cannot set the dtype explicitly. Please use module.to(new_dtype).")

    @property
    def device(self) -> Union[str, torch.device]:
        return self._device

    @device.setter
    def device(self, new_device: Union[str, torch.device]):
        raise RuntimeError("Cannot set the device explicitly. Please use module.to(new_device).")

    def to(self, *args, **kwargs) -> Module:
        out = torch._C._nn._parse_to(*args, **kwargs)
        self.__update_properties(device=out[0], dtype=out[1])
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[int] = None) -> Module:
        self.__update_properties(device=torch.device("cuda", index=device))
        return super().cuda(device=device)

    def cpu(self) -> Module:
        self.__update_properties(device=torch.device("cpu"))
        return super().cpu()

    def type(self, dst_type: Union[str, torch.dtype]) -> Module:
        self.__update_properties(dtype=dst_type)
        return super().type(dst_type=dst_type)

    def float(self) -> Module:
        self.__update_properties(dtype=torch.float)
        return super().float()

    def double(self) -> Module:
        self.__update_properties(dtype=torch.double)
        return super().double()

    def half(self) -> Module:
        self.__update_properties(dtype=torch.half)
        return super().half()

    def __update_properties(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        def apply_fn(module):
            if not isinstance(module, BaseModule):
                return
            if device is not None:
                module._device = device
            if dtype is not None:
                module._dtype = dtype

        self.apply(apply_fn)
