from .pydantic_from_json import pydantic_from_json
from DisPrint import dis_print

def torch_parameters_model(model):
    num = sum(p.numel() for p in model.parameters())
    if num >= 1e9:
        total_parameters= f"{num / 1e9:.3f}B"
    elif num >= 1e6:
        total_parameters= f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        total_parameters= f"{num / 1e3:.3f}K"
    else:
        total_parameters= str(num)
    
    total_layers = sum(1 for _ in model.children())

    return total_parameters,total_layers


def run_index(step,len_files):
    index=0
    run_index=[]
    while index<=len_files:
        lass=index
        index+=100
        run_index.append([lass,len_files if index>=len_files else index])
    
    return run_index


__all__ = [
    "pydantic_from_json",
    "torch_parameters_model",
    "run_index"
]