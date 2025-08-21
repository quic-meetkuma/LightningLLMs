"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-16 11:39:40
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:41:37
# @ Description:
"""

"""
Distributed training utilities for PyTorch and PyTorch Lightning.

This module provides helper functions for distributed training scenarios,
including functions to get process rank, world size, and other distributed
training utilities.
"""

from typing import Any

import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    """
    Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed training is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank() -> int:
    """
    Get the rank of the current process in distributed training.

    Returns:
        int: The rank of the current process. Returns 0 if distributed training
             is not available or initialized.
    """
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    Get the total number of processes in distributed training.

    Returns:
        int: The total number of processes. Returns 1 if distributed training
             is not available or initialized.
    """
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return get_rank() == 0


def sync_across_processes(tensor: torch.Tensor) -> torch.Tensor:
    """
    Synchronize a tensor across all processes by averaging.

    Args:
        tensor (torch.Tensor): The tensor to synchronize.

    Returns:
        torch.Tensor: The synchronized tensor.
    """
    if not is_dist_available_and_initialized():
        return tensor

    # Clone the tensor to avoid modifying the original
    synced_tensor = tensor.clone()
    dist.all_reduce(synced_tensor, op=dist.ReduceOp.SUM)
    synced_tensor /= get_world_size()

    return synced_tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a Python object from source process to all other processes.

    Args:
        obj (Any): The object to broadcast (only used by the source process).
        src (int): The source process rank. Defaults to 0.

    Returns:
        Any: The broadcasted object.
    """
    if not is_dist_available_and_initialized():
        return obj

    # Create a list to store the object on all processes
    if get_rank() == src:
        # Source process: create a list with the object
        obj_list = [obj]
    else:
        # Non-source processes: create an empty list
        obj_list = [None]

    # Broadcast the list
    dist.broadcast_object_list(obj_list, src=src)

    return obj_list[0]


def all_gather_object(obj: Any) -> list:
    """
    Gather Python objects from all processes.

    Args:
        obj (Any): The object to gather from the current process.

    Returns:
        list: A list containing objects from all processes.
    """
    if not is_dist_available_and_initialized():
        return [obj]

    # Create a list to store objects from all processes
    obj_list = [None for _ in range(get_world_size())]

    # Gather objects from all processes
    dist.all_gather_object(obj_list, obj)

    return obj_list


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce a dictionary of tensors across all processes.

    Args:
        input_dict (dict): Dictionary of tensors to reduce.
        average (bool): Whether to average the reduced values. Defaults to True.

    Returns:
        dict: Reduced dictionary (only meaningful on the main process).
    """
    if not is_dist_available_and_initialized():
        return input_dict

    # Flatten the dictionary to a list of tensors and keys
    names = sorted(input_dict.keys())
    values = [input_dict[name] for name in names]

    # Convert all values to tensors if they aren't already
    values = [
        (
            torch.as_tensor(value).cuda()
            if torch.is_tensor(value)
            else torch.as_tensor(value).cuda()
        )
        for value in values
    ]

    # Flatten all tensors
    values = torch.stack(values, dim=0)

    # Reduce across all processes
    dist.all_reduce(values)

    if average:
        values /= get_world_size()

    # Convert back to dictionary
    reduced_dict = {name: value for name, value in zip(names, values)}

    return reduced_dict
