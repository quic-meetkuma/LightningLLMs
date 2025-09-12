
from LightningLLM.components.optimizer import get_optimizer_cls
from LightningLLM.components.callback import get_callback_cls
from peft import LoraConfig, get_peft_model


def get_optimizer(config_manager):
    opt_config = config_manager.get_optimizer_config()
    opt_name = opt_config.pop("optimizer_name")
    opt_cls = get_optimizer_cls(opt_name)
    opt_config['lr'] = float(opt_config['lr'])
    optimizer_cls_and_kwargs = (opt_cls, opt_config)
    return optimizer_cls_and_kwargs

def get_callbacks(config_manager):
    callbacks = []
    callback_config = config_manager.get_callback_config()
    for callback_name, callback_dict in callback_config.items():
        # FIXME: this could have been fixed in config manager.
        if callback_dict is None:
            callback_dict = {}
        callback_inst = get_callback_cls(callback_name)(**callback_dict)
        callbacks.append(callback_inst)
    return callbacks


def prepare_lora(config_manager, model):
    model_config = config_manager.get_model_config()
    use_peft = model_config.get("use_peft", False)
    lora_config = None
    if use_peft:
        peft_config = model_config.get("peft_config", {})

        lora_config = LoraConfig(
            r=peft_config.get("lora_r"),
            lora_alpha=peft_config.get("lora_alpha"),
            target_modules=peft_config.get("target_modules"),
            lora_dropout=peft_config.get("lora_dropout"),
            bias=peft_config.get("bias"),
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, lora_config