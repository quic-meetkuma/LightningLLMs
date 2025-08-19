# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Any, Dict, Union

import torch

try:
    import torch_qaic
except Exception as e:
    print(f"Unable to register qaic backend due to exception: {e}")

import subprocess

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators import _AcceleratorRegistry
from typing_extensions import override
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class QAICAccelerator(Accelerator):
    """Support for a Qualcomm's AI100 Accelerator, optimized for large-scale machine learning."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """Create and prepare the device for the current process."""
        if device.type != "qaic":
            raise MisconfigurationException(
                f"Device should be QAIC, got {device} instead."
            )
        # torch.qaic.set_device(device)

    @override
    def teardown(self) -> None:
        """Clean up any state created by the accelerator."""
        torch.qaic.empty_cache()

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        pass

    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument

        if (
            devices is None
            or (isinstance(devices, int) and devices == 0)
            or str(devices).strip() in ("0", "[]")
        ):
            return None

        devices = _normalize_parse_device_string_input(devices)
        available_devices = _get_available_device_id()
        if not available_devices:
            return None
        if devices == -1:
            return available_devices
        else:
            return available_devices[:devices]

    @staticmethod
    @override
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        return [torch.device("qaic", idx) for idx in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.qaic.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.qaic.is_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return {}

    # @override
    # def get_distribute_name(self) -> str:
    #     return "qccl"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "qaic",
            cls,
            description="QAIC AI100 Accelerator",
        )


def _normalize_parse_device_string_input(
    s: Union[int, str, list[int]],
) -> Union[int, list[int]]:
    if not isinstance(s, str):
        return s
    if s == "-1":
        return -1
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if len(x) > 0]
    return int(s.strip())


def _get_available_device_id():
    """
    API to check available device id.

    Return:
        :int: Available device id.
    """
    return range(torch_qaic.qaic.device_count())
    device_id = 0
    result = None

    available_devices = []
    # FIXME: goes into infinite loop when user doesn't have permission and the command gives permission denied.
    # To reproduce change the ownership of available devices.
    while device_id < torch_qaic.qaic.device_count():
        command = ["/opt/qti-aic/tools/qaic-util", "-q", "-d", f"{device_id}"]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except OSError:
            print("Not a Cloud AI 100 device, Command not found", command)
            return None
        if result:
            if "Status:Ready" in result.stdout:
                print(f"Device {device_id} is available.")
                available_devices.append(device_id)
            else:
                print(f"Device {device_id} is not available.")
            device_id += 1
        else:
            break
    return available_devices
