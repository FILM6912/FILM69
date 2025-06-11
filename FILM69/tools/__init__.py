from .pydantic_from_json import pydantic_from_json
import torch.nn as nn
import torch

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
    ตรวจสอบ input shape ที่โมเดลคาดหวังจากเลเยอร์แรก และคืนค่า shape พร้อมตัวอย่าง tensor
    Args:
        model: PyTorch model (nn.Module)
    Returns:
        tuple: (input_shape_str, example_tensor)
            - input_shape_str: str อธิบาย input shape เช่น "(batch_size, 3, height, width)"
            - example_tensor: torch.Tensor ตัวอย่าง เช่น torch.randn(1, 3, 224, 224)
    """
    first_layer = None
    # หาเลเยอร์แรกที่เป็น nn.Module
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d)):
            first_layer = layer
            break
    
    if first_layer is None:
        return "ไม่พบเลเยอร์ที่ระบุ input shape ได้", None
    
    # ตรวจสอบ device และ dtype จากโมเดล
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # สร้างตัวอย่าง tensor โดยให้สอดคล้องกับ device และ dtype ของโมเดล
    if isinstance(first_layer, nn.Conv2d):
        # สำหรับ Conv2d: input shape = (batch_size, in_channels, height, width)
        shape_str = f"(batch_size, {first_layer.in_channels}, height, width)"
        example = torch.randn(1, first_layer.in_channels, 224, 224, device=device, dtype=dtype)
        return shape_str, example
    elif isinstance(first_layer, nn.Linear):
        # สำหรับ Linear: input shape = (batch_size, in_features)
        shape_str = f"(batch_size, {first_layer.in_features})"
        example = torch.randn(1, first_layer.in_features, device=device, dtype=dtype)
        return shape_str, example
    elif isinstance(first_layer, nn.Conv1d):
        # สำหรับ Conv1d: input shape = (batch_size, in_channels, length)
        shape_str = f"(batch_size, {first_layer.in_channels}, length)"
        example = torch.randn(1, first_layer.in_channels, 128, device=device, dtype=dtype)
        return shape_str, example
    elif isinstance(first_layer, nn.Conv3d):
        # สำหรับ Conv3d: input shape = (batch_size, in_channels, depth, height, width)
        shape_str = f"(batch_size, {first_layer.in_channels}, depth, height, width)"
        example = torch.randn(1, first_layer.in_channels, 16, 224, 224, device=device, dtype=dtype)
        return shape_str, example
    else:
        return "ไม่สามารถระบุ input shape ได้", None
    
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
