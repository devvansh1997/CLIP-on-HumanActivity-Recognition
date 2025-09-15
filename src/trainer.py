import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp import GradScaler, autocast
import time
import os
from typing import Dict, Any

class Trainer:
    """
    The main training engine.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 optimizer: Optimizer,
                 loss_fn: nn.Module,
                 config: Dict[str, Any],
                 run_name: str):
        
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.run_name = run_name
        self.model_type = self.config['model'].get('type', 'clip')

        # Retrieve training parameters from the config
        self.epochs = self.config['training']['epochs']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = self.config['training']['use_amp']
        self.accumulation_steps = self.config['training']['gradient_accumulation_steps']

        self.scaler = GradScaler(self.device, enabled=self.use_amp)
        self.model.to(self.device)
        print(f"Trainer initialized. Running on device: {self.device}")

    def _run_one_epoch(self, epoch: int):
        """Helper function to run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            model_inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model_inputs.pop('label_idx', None)

            with autocast(self.device, enabled=self.use_amp):
                # This is the unified logic for both models
                logits = self.model(**model_inputs)
                loss = self.loss_fn(logits)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            if (i + 1) % (self.accumulation_steps * 10) == 0:
                print(f"  Step [{i+1}/{len(self.train_loader)}], Avg Loss: {total_loss / (i+1):.4f}")
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_epoch_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1}/{self.epochs} completed in {epoch_duration:.2f}s. Average Loss: {avg_epoch_loss:.4f}")

    def train(self):
        """The main training loop."""
        print("🚀 Starting training...")
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
            self._run_one_epoch(epoch)
            self.save_checkpoint(epoch)

        print("✅ Training finished successfully!")

    def save_checkpoint(self, epoch: int):
        """Saves a model and optimizer checkpoint for resumable training."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_filename = f"{self.run_name}_epoch_{epoch+1}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")