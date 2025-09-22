"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 22:59:44
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:36:42
# @ Description:
"""

"""
Model architecture for the modular training system.
Provides BaseModel abstract class and specialized implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer

from LightningLLM.components.component_registry import registry
from LightningLLM.components.logger import get_logger
from LightningLLM.utils.dataset_helper import insert_pad_token

import os
import torch.distributed as dist
from peft import LoraConfig, PeftConfig, get_peft_model

logger = get_logger(__name__)


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models in the training system.
    Provides common interface and functionality for different model types.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base model.

        Args:
            model_name: Name or identifier for the model
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.model_name = model_name
        self.model_config = kwargs
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Load and return the underlying model.

        Returns:
            The loaded model (nn.Module)
        """
        pass

    @abstractmethod
    def load_tokenizer(self) -> Any:
        """
        Load and return the tokenizer.

        Returns:
            The loaded tokenizer
        """
        pass

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        if self._model is None:
            self._model = self.load_model()
        return self._model(*args, **kwargs)

    def get_tokenizer(self):
        """Get the tokenizer for this model."""
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize token embeddings if the model supports it."""
        if self._model is None:
            self._model = self.load_model()

        if hasattr(self._model, "resize_token_embeddings"):
            self._model.resize_token_embeddings(new_num_tokens)
        else:
            logger.log_rank_zero(
                f"Model {self.model_name} does not support token embedding resizing",
                logging.WARNING,
            )

    def get_input_embeddings(self):
        """Get input embeddings if the model supports it."""
        if self._model is None:
            self._model = self.load_model()

        if hasattr(self._model, "get_input_embeddings"):
            return self._model.get_input_embeddings()
        else:
            logger.log_rank_zero(
                f"Model {self.model_name} does not support getting input embeddings",
                logging.WARNING,
            )
            return None

    def parameters(self, recurse: bool = True):
        """Get model parameters."""
        if self._model is None:
            self._model = self.load_model()
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Get named model parameters."""
        if self._model is None:
            self._model = self.load_model()
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        """Get model state dict."""
        if self._model is None:
            self._model = self.load_model()
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load model state dict."""
        if self._model is None:
            self._model = self.load_model()
        return self._model.load_state_dict(state_dict, *args, **kwargs)

    def to(self, device):
        """Move model to device."""
        if self._model is None:
            self._model = self.load_model()
        return self._model.to(device)

    def train(self, mode: bool = True):
        """Set training mode."""
        if self._model is None:
            self._model = self.load_model()
        return self._model.train(mode)

    def eval(self):
        """Set evaluation mode."""
        if self._model is None:
            self._model = self.load_model()
        return self._model.eval()


@registry.model("hf")
class HFModel(BaseModel):
    """
    Hugging Face model wrapper that uses AutoModel classes.
    Supports both causal language models and sequence classification models.
    """

    def __init__(
        self, model_name: str, auto_class_name: str = "AutoModelForCausalLM", **kwargs
    ):
        """
        Initialize HF model.

        Args:
            model_name: Hugging Face model name or path
            auto_class_name: Name of the Auto class of HF. Default is
                "AutoModelForCausalLM".
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, **kwargs)
        self.auto_class_name = auto_class_name
        self.tokenizer_name = kwargs.get("tokenizer_name", model_name)
        self.use_peft = self.model_config.get("use_peft", False)
        self.lora_config = None
        # self.parallel_config = kwargs.get("parallel_config", None)
        if self.use_peft:
            peft_config = self.model_config.get("peft_config", {})

            self.lora_config = LoraConfig(
                r=peft_config.get("lora_r"),
                lora_alpha=peft_config.get("lora_alpha"),
                target_modules=peft_config.get("target_modules"),
                lora_dropout=peft_config.get("lora_dropout"),
                bias=peft_config.get("bias"),
                task_type="CAUSAL_LM",
            )


    def load_model(self) -> nn.Module:
        """Load Hugging Face model based on task type."""
        logger.log_rank_zero(
            f"Loading Hugging Face model: {self.model_name} from class: {self.auto_class_name}"
        )

        if not hasattr(transformers, self.auto_class_name):
            raise ValueError(
                f"Unsupported AutoModel class: {self.auto_class_name}. "
                f"Available classes: {', '.join([name for name in dir(transformers) if name.startswith('AutoModel')])}"
            )
        hf_cls = getattr(transformers, self.auto_class_name)

        quant_config = None
        if self.model_config.get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Optimal for normal distributions
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Saves additional memory
            )


        # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        #     from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
        #     # device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        #     dist.init_process_group("nccl")
        #     rank = dist.get_rank()
        #     world_size = dist.get_world_size()
        #     local_rank = int(os.environ["LOCAL_RANK"])
        #     torch.cuda.set_device(local_rank)
            
            
        #     tp_size = 2
        #     dp_size = 2
        #     cp_size = 1
        #     assert world_size == tp_size * dp_size * cp_size, (
        #         f"World size ({world_size}) must equal TP size ({tp_size}) * DP size ({dp_size}) * CP size ({cp_size})"
        #     )

        #     mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
        #     world_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp", "cp"))
        #     tp_mesh = world_mesh["tp"]
        #     dp_mesh = world_mesh["dp"]
        #     cp_mesh = world_mesh["cp"]
        
        # from accelerate.state import PartialState

        # state = PartialState()

        # Load model based on task type
        model = hf_cls.from_pretrained(
            self.model_name,
            use_cache=self.model_config.get("use_cache"),
            dtype=self.model_config.get("torch_dtype"),
            attn_implementation=self.model_config.get("attn_implementation"),
            device_map=self.model_config.get("device_map"),
            quantization_config=quant_config,
            # tp_size=2,
            # tp_plan="auto", 
            # device_map={"": PartialState().process_index},
            # device_mesh=tp_mesh if dist.is_initialized() else None,
        )

        return model

    def load_tokenizer(self) -> AutoTokenizer:
        """Load Hugging Face tokenizer."""
        logger.log_rank_zero(f"Loading tokenizer: {self.tokenizer_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        insert_pad_token(tokenizer)

        return tokenizer

    def load_peft_config(self) -> PeftConfig:
        """Load the PEFT config."""
        return self.lora_config