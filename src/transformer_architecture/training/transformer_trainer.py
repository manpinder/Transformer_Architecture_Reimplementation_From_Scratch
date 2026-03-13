import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from common.generate_masks import generate_square_subsequent_mask, generate_padding_mask

class TransformerTrainer:
    """Trainer class for training a Transformer model with features like mixed precision, gradient accumulation, \
        early stopping, learning rate scheduling, and gradient clipping.
    Args:
        model: Transformer model instance.
        tokenizer: Tokenizer for input preprocessing.
        device: Device to run training on ('mps', 'cuda', 'cpu').
        experiment_name: Name for experiment directory.
        dataset_name: Name of the dataset being used.
    """
    def __init__(self, model, tokenizer, device, experiment_name, dataset_name):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []

    def setup_experiment_directory(self):
        """Creates directory for experiment results."""
        self.experiment_dir = f"./experiments/{self.experiment_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/checkpoints", exist_ok=True)
        print(f"Experiment directory created: {self.experiment_dir}")

    def initialize_weights(self):
        """Initializes model weights using Xavier initialization."""
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_optimizer(self, learning_rate, weight_decay, betas, eps):
        """Creates Adam optimizer
        Args:
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            betas: Tuple of (beta1, beta2) for Adam optimizer.
            eps: Epsilon for numerical stability in Adam optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

    def get_scheduler(self, optimizer, warmup_steps):
        """Creates a linear warmup scheduler.
        Args:
            optimizer: Optimizer for which to schedule the learning rate.
            warmup_steps: Number of steps for linear warmup.
        """
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_criterion(self, label_smoothing):
        """Creates Cross Entropy loss function
        Args:
            label_smoothing: Amount of label smoothing to apply (0 for no smoothing).
        """
        return nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=label_smoothing
        )

    def create_masks(self, src, tgt_input):
        """Creates source and target masks for attention.
        Args:
            src: Source input tensor.
            tgt_input: Target input tensor (shifted right).
        """
        src_mask = generate_padding_mask(src, self.tokenizer.pad_token_id).to(self.device)
        tgt_subsequent_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
        tgt_padding_mask = generate_padding_mask(tgt_input, self.tokenizer.pad_token_id).to(self.device)
        tgt_mask = tgt_subsequent_mask * tgt_padding_mask
        return src_mask, tgt_padding_mask, tgt_mask

    def train_epoch(self, train_loader, optimizer, scheduler, criterion, scaler, accumulation_steps):
        """Trains for one epoch with mixed precision, gradient accumulation, early stopping, learning rate scheduling, and gradient clipping.
        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for updating model parameters.
            scheduler: Learning rate scheduler.
            criterion: Loss function.
            scaler: GradScaler for mixed precision training.
            accumulation_steps: Number of steps to accumulate gradients before updating model parameters.
        """
        self.model.train()
        total_loss = 0
        total_grad_norm = 0
        accumulation_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for i, batch in enumerate(progress_bar):
            src_data = batch['input_ids'].to(self.device)
            tgt_data = batch['labels'].to(self.device)
            tgt_input = tgt_data[:, :-1]
            tgt_output = tgt_data[:, 1:]
            src_mask, tgt_padding_mask, tgt_mask = self.create_masks(src_data, tgt_input)

            with torch.amp.autocast(device_type=self.device, enabled=scaler is not None):
                output = self.model(src=src_data, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
                loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                loss = loss / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_loss += loss.item()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                total_grad_norm += grad_norm.item()
                total_loss += accumulation_loss * accumulation_steps
                accumulation_loss = 0
                current_lr = scheduler.get_last_lr()[0]

                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'grad_norm': f'{grad_norm:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        return total_loss / len(train_loader), total_grad_norm / max(1, (len(train_loader) // accumulation_steps))

    def validate(self, val_loader, criterion, scaler):
        """Validates the model.
        Args:
            val_loader: DataLoader for validation data.
            criterion: Loss function.
            scaler: GradScaler for mixed precision training.
        """
        self.model.eval()
        total_loss = 0
        start = time.time()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                src_data = batch['input_ids'].to(self.device)
                tgt_data = batch['labels'].to(self.device)
                tgt_input = tgt_data[:, :-1]
                tgt_output = tgt_data[:, 1:]
                src_mask, tgt_padding_mask, tgt_mask = self.create_masks(src_data, tgt_input)

                with torch.amp.autocast(device_type=self.device, enabled=scaler is not None):
                    output = self.model(src=src_data, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
                    loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
                total_loss += loss.item()
        end = time.time()
        return total_loss / len(val_loader), end - start

    def early_stopping(self, val_loss, patience, min_delta):
        """Implements early stopping.
        Args:
            val_loss: Current validation loss.
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in validation loss to qualify as an improvement.
        """
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience

    def save_checkpoint(self, epoch, optimizer, scheduler, scaler, val_loss, is_best):
        """Saves training checkpoint.
        Args:
            epoch: Current epoch number.
            optimizer: Optimizer state to save.
            scheduler: Scheduler state to save.
            scaler: GradScaler state to save (if using mixed precision).
            val_loss: Current validation loss to save.
            is_best: Whether this checkpoint has the best validation loss so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'val_loss': val_loss,
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, f"{self.experiment_dir}/checkpoints/checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(checkpoint, f"{self.experiment_dir}/best_model.pt")

    def log_metrics(self, epoch, train_loss, val_loss, grad_norm, lr, inf_time):
        """Logs training metrics.
        Args:
            epoch: Current epoch number.
            train_loss: Training loss for the epoch.
            val_loss: Validation loss for the epoch.
            grad_norm: Average gradient norm for the epoch.
            lr: Learning rate for the epoch.
            inf_time: Inference time for validation.
        """
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'grad_norm': grad_norm,
            'learning_rate': lr,
            'inference_time': inf_time,
            'timestamp': datetime.now().isoformat()
        }

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        with open(f"{self.experiment_dir}/training_metrics.json", 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def plot_training_history(self):
        """Plots training history."""
        fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(15, 10))
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        ax1.grid(True)
        ax2.plot(self.learning_rates)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        ax3.plot(self.grad_norms)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norms')
        ax3.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()

    def train(self, train_loader, val_loader, epochs, use_mixed_precision, accumulation_steps, 
              patience, min_delta, learning_rate, weight_decay, betas, eps, warmup_steps, label_smoothing):
        """Runs complete training pipeline.
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of epochs to train for.
            use_mixed_precision: Whether to use mixed precision training.
            accumulation_steps: Number of steps to accumulate gradients before updating model parameters.
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in validation loss to qualify as an improvement.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
            betas: Tuple of (beta1, beta2) for Adam optimizer.
            eps: Epsilon for numerical stability in Adam optimizer.
            warmup_steps: Number of steps for linear warmup.
            label_smoothing: Amount of label smoothing to apply (0 for no smoothing).
        """
        self.setup_experiment_directory()
        print(f"Starting training for {epochs} epochs...")  
        self.initialize_weights()
        optimizer = self.get_optimizer(learning_rate, weight_decay, betas, eps)
        scheduler = self.get_scheduler(optimizer, warmup_steps)
        criterion = self.get_criterion(label_smoothing)
        scaler = torch.amp.GradScaler(self.device) if use_mixed_precision else None
        best_val_loss = float('inf')
        inference_times = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            train_loss, avg_grad_norm = self.train_epoch(
                train_loader, optimizer, scheduler, criterion, scaler, accumulation_steps
            )
            val_loss, inf_time = self.validate(val_loader, criterion, scaler)
            inference_times.append(inf_time)
            current_lr = scheduler.get_last_lr()[0]
            self.log_metrics(epoch + 1, train_loss, val_loss, avg_grad_norm, current_lr, inf_time)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            # self.save_checkpoint(epoch + 1, optimizer, scheduler, scaler, val_loss, is_best)
            if self.early_stopping(val_loss, patience, min_delta):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        self.plot_training_history()
        print(f"Training completed! \nBest validation loss: {best_val_loss:.4f}")
        metrics = []
        with open(f"{self.experiment_dir}/training_metrics.json", 'r') as f:
            metrics = [json.loads(line) for line in f]
        return metrics, sum(inference_times) / len(inference_times)