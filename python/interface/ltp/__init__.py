__version__ = "4.2.0"

from .interface import LTP
from ltp_extension.algorithms import StnSplit

__all__ = [
    "LTP",
    "StnSplit",
    "__version__",
]
