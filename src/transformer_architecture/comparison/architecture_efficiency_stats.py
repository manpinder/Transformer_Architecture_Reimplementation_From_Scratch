def get_model_size_mb(model):
    """Calculates the size of the model parameters in MB.
    Args:
        model: The PyTorch model to evaluate.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def count_parameters(model):
    """Counts trainable and total parameters.
    Args:
        model: The PyTorch model to evaluate.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def compare_architectures(custom_model, pytorch_model, custom_avg_inf_time, pytorch_avg_inf_time):
    """
    Runs a full comparison suite and prints a report.
    Args:
        custom_model: The custom PyTorch model to evaluate.
        pytorch_model: The official PyTorch model to evaluate.
        custom_avg_inf_time: Average inference time for the custom model on validation data.
        pytorch_avg_inf_time: Average inference time for the PyTorch model on validation data.
    """
    print(f"--- Model Comparison Report ---")
    
    c_total, c_train = count_parameters(custom_model)
    p_total, p_train = count_parameters(pytorch_model)
    
    print(f"{'Metric':<25} | {'Custom Model':<15} | {'PyTorch Official':<15}")
    print("-" * 65)
    print(f"{'Total Parameters':<25} | {c_total:<15,} | {p_total:<15,}")
    print(f"{'Trainable Parameters':<25} | {c_train:<15,} | {p_train:<15,}")
    
    c_size = get_model_size_mb(custom_model)
    p_size = get_model_size_mb(pytorch_model)
    print(f"{'Model Size (MB)':<25} | {c_size:<15.2f} | {p_size:<15.2f}")
    
    return {
        'custom': {'params': c_total, 'size_mb': c_size, 'inf_time': custom_avg_inf_time},
        'pytorch': {'params': p_total, 'size_mb': p_size, 'inf_time': pytorch_avg_inf_time}
    }