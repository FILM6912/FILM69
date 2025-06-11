from .pydantic_from_json import pydantic_from_json
from .DisPrint import dis_print

def torch_parameters_model(model):
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
    "torch_parameters_model",
    "run_index"
]