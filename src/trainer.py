import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import time
import os
from typing import Dict, Any

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 optimizer: Optimizer,
                 loss_fn: nn.Module,
                 config: Dict[str, Any]):
        
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

        # wandb.watch(self.model, log="all", log_freq=100)
        print(f"Trainer initialized. Running on device: {self.device}")

    def _run_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast(device_type=self.device, enabled=self.use_amp):
                # --- THIS IS THE CORRECTED LINE ---
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_fn(outputs)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # wandb.log({"train_step_loss": loss.item() * self.accumulation_steps})
            
            if (i + 1) % (self.accumulation_steps * 10) == 0:
                print(f"  Step [{i+1}/{len(self.train_loader)}], Avg Loss: {total_loss / (i+1):.4f}")
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_epoch_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1}/{self.epochs} completed in {epoch_duration:.2f}s. Average Loss: {avg_epoch_loss:.4f}")
        

    def train(self):
        print("🚀 Starting training...")
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
            self._run_one_epoch(epoch)
            self.save_checkpoint(epoch)

        print("✅ Training finished successfully!")

    def save_checkpoint(self, epoch: int):
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")