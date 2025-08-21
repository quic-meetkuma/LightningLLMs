"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 22:57:14
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:34:15
# @ Description:
"""

"""
Training Modules for the training system.
"""


from peft import LoraConfig, get_peft_model
from pytorch_lightning import LightningModule

from components.component_registry import ComponentFactory, registry


@registry.trainer_module("causal_lm")
class CausalLMModule(LightningModule):
    """Causal LM module for autoregressive task."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # Load model and tokenizer
        self.model_config = self.config.get_model_config()
        self.use_peft = self.model_config.get("use_peft", False)
        if self.use_peft:
            self.peft_config = self.model_config.get("peft_config", {})

        self.loss_config = self.config.get_loss_config()
        self.scheduler_config = self.config.get_scheduler_config()
        self.opt_config = self.config.get_optimizer_config()
        model_cls = ComponentFactory.create_model(**self.model_config)
        self.model = model_cls.load_model()
        self.tokenizer = model_cls.load_tokenizer()

        ## Apply PEFT here.
        self.apply_lora()

        # Save hyperparameters
        # self.save_hyperparameters()

        # Initialize loss function
        self.loss_fns = []
        for loss_fn_name, loss_fn_dict in self.loss_config["loss_functions"].items():
            loss_fn_inst = ComponentFactory.create_loss_function(
                loss_fn_name, **loss_fn_dict
            )
            self.loss_fns.append(loss_fn_inst)

    def parameters(self):
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                yield parameter

    def apply_lora(self):
        """Apply LoRA to the model if configured."""
        if not self.use_peft:
            return

        lora_config = LoraConfig(
            r=self.peft_config.get("lora_r"),
            lora_alpha=self.peft_config.get("lora_alpha"),
            target_modules=self.peft_config.get("target_modules"),
            lora_dropout=self.peft_config.get("lora_dropout"),
            bias=self.peft_config.get("bias"),
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, **batch_input):
        """Forward pass through the model."""
        output = self.model(**batch_input)
        return output

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = 0
        for loss_inst in self.loss_fns:
            loss += loss_inst(outputs=output.logits, targets=labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = 0
        for loss_inst in self.loss_fns:
            loss += loss_inst(logits=output.logits, labels=labels)

        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler."""
        max_steps = self.trainer.estimated_stepping_batches
        self.scheduler_config["num_training_steps"] = max_steps

        optimizers = []
        schedulers = []
        for opt_name, opt_dict in self.opt_config["optimizers"].items():
            opt_inst = ComponentFactory.create_optimizer(
                opt_name, **opt_dict, model_params=self.parameters()
            )
            optimizers.append(opt_inst.optimizer)
            scheduler = ComponentFactory.create_scheduler(
                **self.scheduler_config,
                optimizer=opt_inst.optimizer,
            )
            schedulers.append(scheduler)

        assert len(schedulers) != 0, "More than 1 schedulers are not supported."
        assert len(optimizers) != 0, "More than 1 optimizers are not supported."
        return {
            "optimizer": optimizers[0],
            "lr_scheduler": {
                "scheduler": schedulers[0],
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, optimizer, metric=None):
        scheduler.step()

    def generate_summary(self, input_text, max_length=128):
        """Generate summary for input text."""
        # Tokenize input text
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.data.max_source_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate summary
        summary_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


@registry.trainer_module("causal_lm_kd")
class CausalLMKDModule(LightningModule):
    """Causal LM module for knowledge distillation and autoregressive task."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # Load model and tokenizer
        import pdb

        pdb.set_trace()
        self.student_model_config = self.config.get_model_config()
        training_config = self.config.get_training_config()
        self.teacher_model_config = training_config.get("teacher_model", None)
        if self.teacher_model_config is None:
            raise RuntimeError(
                "teacher_model key missing under training section of config."
            )
        self.use_peft = self.model_config.get("use_peft", False)
        if self.use_peft:
            self.peft_config = self.model_config.get("peft_config", {})

        self.loss_config = self.config.get_loss_config()
        self.scheduler_config = self.config.get_scheduler_config()
        self.opt_config = self.config.get_optimizer_config()
        model_cls = ComponentFactory.create_model(**self.model_config)
        self.model = model_cls.load_model()
        self.tokenizer = model_cls.load_tokenizer()

        ## Apply PEFT here.
        self.apply_lora()

        # Save hyperparameters
        # self.save_hyperparameters()

        # Initialize loss function
        self.loss_fn = ComponentFactory.create_loss_function(**self.loss_config)

    def parameters(self):
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                yield parameter

    def apply_lora(self):
        """Apply LoRA to the model if configured."""
        if not self.use_peft:
            return

        lora_config = LoraConfig(
            r=self.peft_config.get("lora_r"),
            lora_alpha=self.peft_config.get("lora_alpha"),
            target_modules=self.peft_config.get("target_modules"),
            lora_dropout=self.peft_config.get("lora_dropout"),
            bias=self.peft_config.get("bias"),
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, **batch_input):
        """Forward pass through the model."""
        output = self.model(**batch_input)
        return output

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = self.loss_fn(output.logits, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = self.loss_fn(output.logits, labels)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler."""
        optimizer = ComponentFactory.create_optimizer(
            **self.opt_config, model_params=self.parameters()
        )

        max_steps = self.trainer.estimated_stepping_batches
        self.scheduler_config["num_training_steps"] = max_steps
        scheduler = ComponentFactory.create_scheduler(
            **self.scheduler_config,
            optimizer=optimizer.optimizer,
        )
        return {
            "optimizer": optimizer.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, optimizer, metric=None):
        scheduler.step()

    def generate_summary(self, input_text, max_length=128):
        """Generate summary for input text."""
        # Tokenize input text
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.data.max_source_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate summary
        summary_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
