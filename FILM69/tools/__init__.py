from .pydantic_from_json import pydantic_from_json
import torch.nn as nn

def torch_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def format_number(num):
        if num >= 1_000_000_000:  # Billion
            return f"{num / 1_000_000_000:.1f}B"
        elif num >= 1_000_000:   # Million
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:       # Thousand
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)
    
    total_layers = sum(1 for _ in model.children())
    
    return {
        'total': format_number(total_params),
        "total_layers":total_layers,
        'trainable': format_number(trainable_params),
        'total_raw': total_params,
        'trainable_raw': trainable_params
    }
    

def torch_get_input_shape(model):
    """
    ตรวจสอบ input shape ที่โมเดลคาดหวังจากเลเยอร์แรก
    Args:
        model: PyTorch model (nn.Module)
    Returns:
        tuple: คำอธิบาย input shape ที่คาดหวัง
    """
    first_layer = None
    # หาเลเยอร์แรกที่เป็น nn.Module
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d)):
            first_layer = layer
            break
    
    if first_layer is None:
        return "ไม่พบเลเยอร์ที่ระบุ input shape ได้"
    
    if isinstance(first_layer, nn.Conv2d):
        # สำหรับ Conv2d: input shape = (batch_size, in_channels, height, width)
        return f"(batch_size, {first_layer.in_channels}, height, width)"
    elif isinstance(first_layer, nn.Linear):
        # สำหรับ Linear: input shape = (batch_size, in_features)
        return f"(batch_size, {first_layer.in_features})"
    elif isinstance(first_layer, nn.Conv1d):
        # สำหรับ Conv1d: input shape = (batch_size, in_channels, length)
        return f"(batch_size, {first_layer.in_channels}, length)"
    elif isinstance(first_layer, nn.Conv3d):
        # สำหรับ Conv3d: input shape = (batch_size, in_channels, depth, height, width)
        return f"(batch_size, {first_layer.in_channels}, depth, height, width)"
    else:
        return "ไม่สามารถระบุ input shape ได้"

def run_index(step,len_files):
    index=0
    run_index=[]
    while index<=len_files:
        lass=index
        index+=step
        run_index.append([lass,len_files if index>=len_files else index])
    
    return run_index


__all__ = [
    "pydantic_from_json",
    "torch_parameters",
    "run_index",
    "torch_get_input_shape"
]
