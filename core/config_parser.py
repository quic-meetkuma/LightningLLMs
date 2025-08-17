"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-13 17:48:51
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:38:17
# @ Description:
"""

"""
Refactored finetuning entry point using the new modular architecture.
This provides a clean, extensible interface for fine-tuning models.
"""

import argparse
import logging
import sys

from core.config_manager import ConfigManager
from components.logger import get_logger
from components.component_registry import ComponentFactory

logger = get_logger()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the new finetuning interface."""
    parser = argparse.ArgumentParser(
        description="Modular fine-tuning system for language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)",
    )

    return parser


def main():
    """Main entry point for the refactored finetuning system."""
    parser = create_parser()
    args = parser.parse_args()

    # Create configuration manager
    config_manager = ConfigManager()
    config_manager.load_config(args.config)

    # Validate configuration
    try:
        config_manager.validate_config()
    except ValueError as e:
        logger.log_rank_zero(f"Configuration validation failed: {e}", logging.ERROR)
        sys.exit(1)

    # Setup logging
    logger.prepare_for_logs(
        config_manager.output_dir, config_manager.save_metrics, config_manager.log_level
    )

    # Log configuration
    logger.log_rank_zero("Starting fine-tuning with configuration:")
    logger.log_rank_zero(f"Model: {config_manager.model.name}")
    logger.log_rank_zero(f"Dataset: {config_manager.dataset.name}")
    logger.log_rank_zero(f"Optimizer: {config_manager.optimizer.name}")
    logger.log_rank_zero(f"Scheduler: {config_manager.scheduler.name}")
    logger.log_rank_zero(f"Device: {config_manager.device}")
    logger.log_rank_zero(f"Batch size: {config_manager.train_batch_size}")
    logger.log_rank_zero(f"Epochs: {config_manager.num_epochs}")

    # Create trainer and start training
    try:
        trainer = ComponentFactory.create_trainer(config_manager)
        results = trainer.train()

        # Log final results
        logger.log_rank_zero("Training completed successfully!")
        logger.log_rank_zero(f"Best validation loss: {results['best_eval_loss']:.4f}")
        logger.log_rank_zero(f"Final epoch: {results['final_epoch']}")
        logger.log_rank_zero(f"Final step: {results['final_step']}")

        return results

    except Exception as e:
        logger.log_rank_zero(f"Training failed with error: {e}", logging.ERROR)
        raise


if __name__ == "__main__":
    main()
