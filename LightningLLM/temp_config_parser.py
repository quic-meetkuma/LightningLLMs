
from transformers.hf_argparser import HfArgumentParser
import sys

from LightningLLM.components.config_manager import OptimizerConfig, SchedulerConfig, DatasetConfig, ModelConfig, CallbackConfig, TrainingConfig, MasterConfig
import os

if __name__ == '__main__':
    parser = HfArgumentParser((MasterConfig))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        master_config = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))[0]
        import pdb; pdb.set_trace()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
