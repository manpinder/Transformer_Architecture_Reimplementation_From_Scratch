import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from comparison.architecture_efficiency_stats import compare_architectures

class ComparisonPlotter:
    """
    Class responsible for plotting the training comparison and architecture efficiency comparison between the custom and 
    PyTorch implementations of the Transformer model. It generates visualizations for training/validation loss, gradient norms, 
    and metrics like parameter count, model size, and inference time.
    Args:
        save_dir: Base directory where the comparison plots will be saved.
        dataset_name: Name of the dataset used for training.
    """

    def __init__(self, save_dir, dataset_name):
        self.experiment_path = None
        self.setup_experiment_directory(save_dir, dataset_name)

    def setup_experiment_directory(self, save_dir, dataset_name):
        """Creates a unique directory for the current experiment based on the dataset name and current timestamp.
        Args:
            save_dir: Base directory where the experiment directory will be created.
            dataset_name: Name of the dataset used for training.
        """
        self.experiment_path = f"{save_dir}/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_path, exist_ok=True)
        
    def plot_training_comparison(self, custom_metrics, pytorch_metrics):
        """
        Plots a comparison between Custom and PyTorch implementation training metrics.
        
        Args:
            custom_metrics: List of dicts from custom model training.
            pytorch_metrics: List of dicts from pytorch model training.
        """
        epochs = [m['epoch'] for m in custom_metrics]
        
        c_train_loss = [m['train_loss'] for m in custom_metrics]
        c_val_loss = [m['val_loss'] for m in custom_metrics]
        c_grad_norm = [m['grad_norm'] for m in custom_metrics]
        
        p_train_loss = [m['train_loss'] for m in pytorch_metrics]
        p_val_loss = [m['val_loss'] for m in pytorch_metrics]
        p_grad_norm = [m['grad_norm'] for m in pytorch_metrics]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
        
        # --- Plot 1: Training Loss Comparison ---
        ax1.plot(epochs, c_train_loss, label='Custom Model', color='blue', linestyle='-', marker='o', markersize=4)
        ax1.plot(epochs, p_train_loss, label='PyTorch Official', color='orange', linestyle='--', marker='x', markersize=4)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Validation Loss Comparison ---
        ax2.plot(epochs, c_val_loss, label='Custom Model', color='blue', linestyle='-', marker='o', markersize=4)
        ax2.plot(epochs, p_val_loss, label='PyTorch Official', color='orange', linestyle='--', marker='x', markersize=4)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Plot 3: Gradient Norm Comparison ---
        ax3.plot(epochs, c_grad_norm, label='Custom Model', color='blue', linestyle='-', marker='o', markersize=4, alpha=0.7)
        ax3.plot(epochs, p_grad_norm, label='PyTorch Official', color='orange', linestyle='--', marker='x', markersize=4, alpha=0.7)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Gradient Norm')  
        ax3.set_title('Gradient Norm Stability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{self.experiment_path}/training_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training comparison plot saved to: {self.experiment_path}/training_comparison.png")
        plt.close()

    def plot_architecture_comparison(self, custom_model, torch_model, custom_avg_inf_time, torch_avg_inf_time):
        """
        Plots architecture comparison metrics (Params, Size, Speed) side-by-side.
        Args:
            custom_model: The custom Transformer model instance.
            torch_model: The PyTorch official Transformer model instance.
            custom_avg_inf_time: Average inference time for the custom model on validation data.
            torch_avg_inf_time: Average inference time for the PyTorch model on validation data.
        """
        metrics = ['Params (M)', 'Size (MB)', 'Val Inference Time (sec)']
        comparison_data = compare_architectures(custom_model, torch_model, custom_avg_inf_time, torch_avg_inf_time)
        
        c_vals = [
            comparison_data['custom']['params'] / 1e6,
            comparison_data['custom']['size_mb'],
            comparison_data['custom']['inf_time']
        ]
        
        p_vals = [
            comparison_data['pytorch']['params'] / 1e6,
            comparison_data['pytorch']['size_mb'],
            comparison_data['pytorch']['inf_time']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, c_vals, width, label='Custom', color='blue', alpha=0.7)
        rects2 = ax.bar(x + width/2, p_vals, width, label='PyTorch', color='orange', alpha=0.7)
        
        ax.set_ylabel('Value')
        ax.set_title('Architecture Efficiency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        ax.bar_label(rects1, padding=3, fmt='%.3f')
        ax.bar_label(rects2, padding=3, fmt='%.3f')
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/architecture_comparison.png", dpi=300)
        print(f"Architecture comparison saved to {self.experiment_path}/architecture_comparison.png")