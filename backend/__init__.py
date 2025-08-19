# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
from pytorch_lightning.accelerators.accelerator import Accelerator
from lightning_fabric.utilities.registry import _register_classes
from pytorch_lightning.accelerators import AcceleratorRegistry

from backend.qaic_accelerator import QAICAccelerator

_register_classes(
    AcceleratorRegistry, "register_accelerators", sys.modules[__name__], Accelerator
)
