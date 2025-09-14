import argparse
import torch
import wandb
from typing import Dict, Any

from src.utils import load_config
from src.dataset import create_dataloader
from src.model import create_model
from src.loss import create_loss_fn
from src.trainer import Trainer

def main(config_path: str):
    """
    Main function to set up and run the training pipeline with robust W&B integration.
    """
    # Load the configuration file
    config = load_config(config_path)

    # --- NEW: Defensive W&B Configuration and Debugging ---
    
    # 1. Explicitly get the W&B config dictionary
    wandb_config = config.get('wandb')
    
    # 2. Add defensive checks to ensure the config is valid
    if not wandb_config:
        raise ValueError("Your config file is missing the 'wandb' section.")
        
    entity = wandb_config.get('entity')
    project = wandb_config.get('project')
    
    if not entity or not project:
        raise ValueError("The 'entity' and 'project' keys must be set in the 'wandb' section of your config.")

    # 3. Add a debug print to show exactly what is being used
    print(f"--- Attempting to initialize W&B with Entity: '{entity}' and Project: '{project}' ---")

    # Initialize Weights & Biases with the validated config
    wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=f"{config['training']['model_type']}-bs{config['training']['mini_batch_size'] * config['training']['gradient_accumulation_steps']}-{config_path.split('/')[-1].replace('.yaml', '')}"
    )
    
    print("--- W&B Initialized Successfully ---")
    
    # The rest of the script remains the same
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
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a CLIP-based model for Human Activity Recognition.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    main(args.config)