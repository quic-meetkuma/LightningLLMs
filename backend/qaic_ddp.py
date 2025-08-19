# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from pytorch_lightning.strategies.ddp import DDPStrategy

from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
import logging

log = logging.getLogger(__name__)
from typing_extensions import override


class QaicDDPStrategy(DDPStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qaic_enabled = False

    @override
    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        device_ids = self.determine_ddp_device_ids()
        log.debug(
            f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}"
        )
        # https://pytorch.org/docs/stable/notes/cuda.html#id5
        return DistributedDataParallel(
            module=model, device_ids=device_ids, **self._ddp_kwargs
        )
