"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-16 11:58:28
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:58:30
# @ Description:
"""

from abc import ABC, abstractmethod

from typing import Type

from LightningLLM.components.component_registry import registry
from transformers.integrations.integration_utils import TensorBoardCallback
from transformers import EarlyStoppingCallback, ProgressCallback, PrinterCallback, DefaultFlowCallback
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers import TrainingArguments
from LightningLLM.utils.qaic_profiler_utils import get_op_verifier_ctx, init_qaic_profiling, stop_qaic_profiling
import os
from functools import partial

registry.callback("early_stopping")(EarlyStoppingCallback)
registry.callback("printer")(PrinterCallback)
registry.callback("default_flow")(DefaultFlowCallback)
registry.callback("tensorboard")(TensorBoardCallback)


@registry.callback("enhanced_progressbar")
class EnhancedProgressCallback(ProgressCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        super().__init__(*args, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.training_bar.set_description("Training Progress")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            
            updated_dict = {}
            if "epoch" in shallow_logs:
                updated_dict["epoch"] = shallow_logs["epoch"]
            if "loss" in shallow_logs:
                updated_dict["loss"] = shallow_logs["loss"]
            if "learning_rate" in shallow_logs:
                updated_dict["lr"] = shallow_logs["learning_rate"]
            self.training_bar.set_postfix(updated_dict)


@registry.callback("qaic_profiler_callback")
class QAICProfilerCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        self.start_step = kwargs.get("start_step", -1)
        self.end_step = kwargs.get("end_step", -1)
        self.device_ids = kwargs.get("device_ids", [0])
        
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if state.global_step == self.start_step:
            for device_id in self.device_ids:
                init_qaic_profiling(True, f"qaic:{device_id}")
        elif state.global_step == self.end_step:
            for device_id in self.device_ids:
                stop_qaic_profiling(True, f"qaic:{device_id}")
    

@registry.callback("qaic_op_by_op_verifier_callback")
class QAICOpByOpVerifierCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        self.start_step = kwargs.get("start_step", -1)
        self.end_step = kwargs.get("end_step", -1)
        self.trace_dir = kwargs.get("trace_dir", "qaic_op_by_op_traces")
        self.atol = kwargs.get("atol", 1e-1)
        self.rtol = kwargs.get("rtol", 1e-5)
        
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.start_step <= state.global_step < self.end_step:
            self.op_verifier_ctx_step = get_op_verifier_ctx(use_op_by_op_verifier=True, device_type="qaic", dump_dir=self.trace_dir, step=state.global_step, atol=self.atol, rtol=self.rtol)
            self.op_verifier_ctx_step.__enter__()
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.start_step <= state.global_step < self.end_step:
            if self.op_verifier_ctx_step is not None:
                self.op_verifier_ctx_step.__exit__(None, None, None)


# default_callbacks = [DefaultFlowCallback(), ProgressCallback()]
def get_callback_cls(callback_name: str) -> type[TrainerCallback]:
    callback_cls = registry.get_callback(callback_name)
    if callback_cls is None:
        raise ValueError(f"Unknown optimizer: {callback_name}")
    return callback_cls
