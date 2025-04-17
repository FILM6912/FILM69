from pydantic_from_json import pydantic_from_json

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

