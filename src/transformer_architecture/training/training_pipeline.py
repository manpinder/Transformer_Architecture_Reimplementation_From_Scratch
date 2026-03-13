from custom_implementation.architecture import CustomTransformer
from pytorch_official_module.architecture import TorchTransformer
from .transformer_trainer import TransformerTrainer

class TrainingPipeline:
    """Main training pipeline that orchestrates the training of both the custom and PyTorch implementations of the Transformer model. 
    Args:
        cfg: Configuration dictionary loaded from YAML file.
        vocab_size: Size of the vocabulary for source and target languages.
        tokenizer: Tokenizer for encoding and decoding text.
        device: Device to run the training on (e.g., 'cpu', 'cuda', 'mps').
        dataset_name: Name of the dataset being used for training.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
    """

    def __init__(self, cfg, tokenizer, device, dataset_name, train_loader, val_loader):
        self.device = device
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.val_loader = val_loader   

    def custom_training_pipeline(self):
        """Sets up the custom Transformer model and training pipeline."""
        custom_model = CustomTransformer(
            len(self.tokenizer), len(self.tokenizer), d_model=self.cfg.get('d_model',512),
            n_heads=self.cfg.get('n_heads',8), n_layers=self.cfg.get('n_layers',6),
            d_ff=self.cfg.get('d_ff',2048), dropout_rate=self.cfg.get('dropout_rate',0.1),
            max_seq_len=self.cfg.get('max_seq_len',128)
        ).to(self.device)
        pipeline = TransformerTrainer(custom_model, self.tokenizer, self.device, 'custom', self.dataset_name)
        return custom_model, pipeline
    
    def pytorch_training_pipeline(self):
        """Sets up the PyTorch Transformer model and training pipeline."""
        torch_model = TorchTransformer(
            len(self.tokenizer), len(self.tokenizer), d_model=self.cfg.get('d_model',512),
            n_heads=self.cfg.get('n_heads',8), n_layers=self.cfg.get('n_layers',6),
            d_ff=self.cfg.get('d_ff',2048), dropout_rate=self.cfg.get('dropout_rate',0.1),
            max_seq_len=self.cfg.get('max_seq_len',128)
        ).to(self.device)
        pipeline = TransformerTrainer(torch_model, self.tokenizer, self.device, 'pytorch', self.dataset_name)
        return torch_model, pipeline
    
    def train(self, model_type):
        """Main method to train the specified model type ('custom' or 'pytorch')."""
        if model_type == 'custom':
            model, pipeline = self.custom_training_pipeline()
        elif model_type == 'pytorch':
            model, pipeline = self.pytorch_training_pipeline()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        metrics, avg_inf_time  = pipeline.train(
            self.train_loader, self.val_loader,
            epochs=self.cfg.get('epochs',10),
            use_mixed_precision=self.cfg.get('use_mixed_precision', True),
            accumulation_steps=self.cfg.get('accumulation_steps',4),
            patience=self.cfg.get('patience',3),
            min_delta=self.cfg.get('min_delta',0.01),
            learning_rate=self.cfg.get('learning_rate',0.0001),
            weight_decay=self.cfg.get('weight_decay',0.01),
            betas=self.cfg.get('betas',(0.9, 0.98)),
            eps=self.cfg.get('eps',1e-9),
            warmup_steps=self.cfg.get('warmup_steps',1000),
            label_smoothing=self.cfg.get('label_smoothing',0.1)
        )
        return model, metrics, avg_inf_time