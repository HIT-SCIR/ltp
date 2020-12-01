from types import ModuleType
from typing import List


class LazyModule(ModuleType):
    requirements: List[str]

    def __init__(self, name, requirements=None):
        super().__init__(name)

        if requirements is None:
            from ltp.const import requirements
            requirements = requirements
        self.name = name
        self.requirement = requirements

    def __getattr__(self, item):
        print(f"Need Install {' '.join(self.requirement)}!!!")
        print(f"pip install {' '.join(self.requirement)}")

        if item == '__path__':
            return []

        return LazyModule(item)


class FakePytorchLightning(ModuleType):
    requirements: List[str]

    def __init__(self, name, requirements=None):
        super().__init__(name)

        if requirements is None:
            from ltp.const import requirements
            requirements = requirements
        self.name = name
        self.requirement = requirements

    def __getattr__(self, item):
        if item == '__path__':
            return []

        return FakePytorchLightning(item)


def fake_import_pytorch_lightning():
    try:
        import pytorch_lightning
    except:
        import sys
        pytorch_lightning = FakePytorchLightning('pytorch_lightning')
        pytorch_lightning_utilities = FakePytorchLightning('pytorch_lightning.utilities')
        pytorch_lightning_utilities_argparse_utils = FakePytorchLightning('pytorch_lightning.utilities.argparse_utils')

        sys.modules['pytorch_lightning'] = pytorch_lightning
        sys.modules['pytorch_lightning.utilities'] = pytorch_lightning_utilities
        sys.modules['pytorch_lightning.utilities.argparse_utils'] = pytorch_lightning_utilities_argparse_utils
