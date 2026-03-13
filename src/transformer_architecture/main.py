import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import yaml
import sys
import pytest
from data.data_loader import create_data_loaders
from training.training_pipeline import TrainingPipeline
from comparison.plot_comparison import ComparisonPlotter

def run_tests():
    """Runs all unit tests using pytest and returns True if all pass, False otherwise."""
    print("Running unit tests...")
    result = pytest.main(["tests", "--verbose"])
    if result == 0: 
        print("All tests passed successfully!")
        return True
    else:
        print("Test failures detected. Aborting training.")
        return False

def device_setup():
    """Determines the best available device for training."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def main():
    """Main function to set up and run the training pipeline."""
    device = device_setup()
    print('Using device:', device)

    if not run_tests():
        print("Exiting due to test failures.")
        sys.exit(1)

    cfg = {}
    try:
        with open('config.yaml') as fh:
            cfg = yaml.safe_load(fh)
    except Exception as e:
        print('Failed to load YAML config', e)

    train_loader, val_loader, tokenizer = create_data_loaders(cfg.get('dataset_name', 'bentrevett/multi30k'), 
                                                    cfg.get('input_column', 'en'), cfg.get('target_column', 'de'),
                                                    cfg.get('n_samples', 10000), cfg.get('tokenizer_name', 't5-small'), 
                                                    cfg.get('max_length', 128), cfg.get('truncation', True), 
                                                    cfg.get('padding', 'max_length'), cfg.get('batch_size', 8),
                                                    cfg.get('val_split', 0.25)
                                                    )
                                                              
    dataset_name = cfg.get('dataset_name', 'bentrevett_multi30k').replace('/', '_')
    pipeline = TrainingPipeline(cfg, tokenizer, device, dataset_name, train_loader, val_loader)
    
    print(">> Training Custom implementation of the Transformer model")
    custom_model, custom_metrics, custom_avg_inf_time = pipeline.train('custom')

    print("\n>> Training PyTorch official implementation of the Transformer model")
    torch_model, torch_metrics, torch_avg_inf_time = pipeline.train('pytorch')

    plotter = ComparisonPlotter(cfg.get('comparison_save_dir', './experiments/comparison'), dataset_name)
    plotter.plot_training_comparison(custom_metrics, torch_metrics)
    plotter.plot_architecture_comparison(custom_model, torch_model, custom_avg_inf_time, torch_avg_inf_time)

if __name__ == '__main__':
    main()
