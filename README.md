# LightningLLMs
Repo provides structured code to train/finetune LLMs on top of Pytorch Lightning.


# Run SFT using LoRA on single device
```python
QAIC_VISIBLE_DEVICES=33 python -m LightningLLM.main ./configs/sft_config_single_device.yaml
```

# Run SFT using LoRA with 8xDDP (via torchrun)
```python
QAIC_VISIBLE_DEVICES=33,34,35,36,37,38,39,40 torchrun --nproc-per-node 8 -m LightningLLM.main ./configs/sft_config_ddp.yaml
```
# Run SFT using LoRA with 8xDDP (via accelerate)
https://github.com/huggingface/accelerate/blob/14383311c22bfa9c67714ec481e94ceec62e6c86/examples/README.md#simple-vision-example
```python
QAIC_VISIBLE_DEVICES=33,34,35,36,37,38,39,40 accelerate launch --config_file ./configs/accelerate/fsdp_config.yaml -m LightningLLM.main ./configs/sft_config_single_device.yaml
```
# Run SFT using LoRA with FSDP (via accelerate)
QAIC_VISIBLE_DEVICES=33,34,35,36,37,38,39,40 accelerate launch --config_file ./configs/accelerate/fsdp_config.yaml -m LightningLLM.main ./configs/sft_config_single_device.yaml

Note: With above FSDP command it is failing for QAIC with below error.
Error: RuntimeError: No backend type associated with device type cpu

# Training Time Benchmarks

## Old QEff stack v/s HF Trainer based stack on same config.

## HF Trainer based stack on various use cases
- Samsum dataset
- Alpaca dataset
- Style Remix dataset
- GSM8k dataset

#