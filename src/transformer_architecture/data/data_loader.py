import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def create_data_loaders(dataset_name: str, input_column: str, target_column: str, n_samples: int, tokenizer_name: str, 
                        max_length: int, truncation: bool, padding: str, batch_size: int, val_split: float):
    """Creates data loaders for training and validation.
    Args:
        dataset_name: The name of the dataset to load.
        input_column: The name of the input column in the dataset.
        target_column: The name of the target column in the dataset.
        n_samples: The number of samples to use from the dataset.
        tokenizer_name: The name of the tokenizer to use.
        max_length: The maximum length of input sequences.
        truncation: Whether to truncate sequences longer than max_length.
        padding: Padding strategy for sequences shorter than max_length.
        batch_size: The batch size for the data loaders.
        val_split: The fraction of the dataset to use for validation.
    """
    dataset = load_dataset(dataset_name, split=f'train[:{n_samples}]')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def preprocess_function(examples):
        """Tokenizes the input texts and summaries."""
        model_inputs = tokenizer(
            [text for text in examples[input_column]],
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        labels = tokenizer(
            [text for text in examples[target_column]],
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=val_split)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    def collate_fn(batch):
        """Collate function to combine samples into a batch."""
        return {
            'input_ids': torch.tensor([item['input_ids'] for item in batch], dtype=torch.long),
            'attention_mask': torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long),
            'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        }

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, tokenizer