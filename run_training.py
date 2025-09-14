import argparse
import torch
import wandb # Import wandb
from typing import Dict, Any

from src.utils import load_config
from src.dataset import create_dataloader
from src.model import create_model
from src.loss import create_loss_fn
from src.trainer import Trainer

def main(config_path: str):
    """
    Main function to set up and run the training pipeline.
    """
    # Load the configuration file
    config = load_config(config_path)

    # --- NEW: Initialize Weights & Biases ---
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        config=config, # This logs all hyperparameters from your config file
        name=f"{config['training']['model_type']}-bs{config['training']['mini_batch_size'] * config['training']['gradient_accumulation_steps']}-{config_path.split('/')[-1].replace('.yaml', '')}"
    )
    
    train_loader = create_dataloader(config, split='train')
    model = create_model(config)
    loss_fn = create_loss_fn(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config
    )
    
    trainer.train()
    
    # --- NEW: Finish the W&B run ---
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a CLIP-based model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config.")
    args = parser.parse_args()
    
    main(args.config)