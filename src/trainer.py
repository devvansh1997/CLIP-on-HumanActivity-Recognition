import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import time
import os
import wandb
from typing import Dict, Any

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 optimizer: Optimizer,
                 loss_fn: nn.Module,
                 config: Dict[str, Any]):
        
        # ... (same initialization code as before) ...
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.epochs = self.config['training']['epochs']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = self.config['training']['use_amp']
        self.accumulation_steps = self.config['training']['gradient_accumulation_steps']
        self.scaler = GradScaler(self.device, enabled=self.use_amp)
        self.model.to(self.device)

        # --- NEW: Watch the model with W&B ---
        # This will log gradients, parameters, and model topology
        wandb.watch(self.model, log="all", log_freq=100)
        
        print(f"Trainer initialized. Running on device: {self.device}")

    def _run_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            # ... (same batch setup code) ...
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast(device_type=self.device, enabled=self.use_amp):
                outputs = self.model(...) # same forward pass
                loss = self.loss_fn(outputs)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # --- NEW: Log step-level loss ---
                wandb.log({"train_step_loss": loss.item() * self.accumulation_steps})

        # ... (same end-of-epoch print statements) ...
        avg_epoch_loss = total_loss / len(self.train_loader)
        
        # --- NEW: Log epoch-level metrics ---
        wandb.log({
            "epoch": epoch,
            "avg_epoch_loss": avg_epoch_loss,
            "epoch_duration_secs": time.time() - start_time
        })

    def train(self):
        # ... (same train method as before) ...
        print("Starting training with W&B tracking...")
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
            self._run_one_epoch(epoch)
            self.save_checkpoint(epoch)
        print("Training finished successfully!")

    def save_checkpoint(self, epoch: int):
        """Saves a model and optimizer checkpoint for resumable training."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        
        # Save both model and optimizer states
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")