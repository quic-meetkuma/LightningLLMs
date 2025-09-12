import importlib.machinery
import importlib.metadata
import importlib.util
import json
import operator
import os
import re
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from enum import Enum
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Callable, Optional, Union

from packaging import version

from . import logging
from transformers.utils.import_utils import _torch_available

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@lru_cache
def is_torch_qaic_available(check_device=False):
    "Checks if `torch_qaic` is installed and potentially if a QAIC is in the environment"
    if not _torch_available or importlib.util.find_spec("torch_qaic") is None:
        return False

    import torch
    import torch_qaic  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.qaic.device_count()
            return torch.qaic.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "qaic") and torch.qaic.is_available()
