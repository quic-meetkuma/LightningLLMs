'''
 # @ Author: Meet Patel
 # @ Create Time: 2025-08-12 17:22:21
 # @ Modified by: Meet Patel
 # @ Modified time: 2025-08-18 00:21:11
 # @ Description:
 '''

"""
Main entry point for fine-tuning LLMs on the SAMSum dataset.
"""

from pathlib import Path

from src.trainer import create_trainer
from core.config_manager import ConfigManager
from core.config_parser import create_parser
from components.dataset import GenericDataModule
from components.component_registry import ComponentFactory


def main():
    """Main entry point for training."""
    # Parse command line arguments using the new parser
    parser = create_parser()
    args = parser.parse_args()

    # Create configuration manager
    config_manager = ConfigManager()
    config_manager.load_config(args.config)

    # Validate configuration
    config_manager.validate_config()

    # Set output directory
    output_dir = config_manager.training.output_dir

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize data module
    dataset_config = config_manager.get_dataset_config()
    data_module = GenericDataModule(dataset_config)

    # Initialize model
    trainer_config = config_manager.get_training_config()
    trainer_type = trainer_config.get("type")
    output_dir = trainer_config.get("output_dir")
    trainer_module = ComponentFactory.create_trainer_module(
        trainer_type, config=config_manager
    )

    # Create trainer
    trainer = create_trainer(config_manager, output_dir)

    # Train model
    trainer.fit(trainer_module, data_module)

    # Test model
    trainer.test(trainer_module, data_module)


if __name__ == "__main__":
    main()
