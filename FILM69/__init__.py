def torch_parameters_model(model):
    num = sum(p.numel() for p in model.parameters())
    if num >= 1e9:
        return f"{num / 1e9:.3f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.3f}K"
    else:
        return str(num)
