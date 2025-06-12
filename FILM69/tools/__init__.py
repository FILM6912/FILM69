from .pydantic_from_json import pydantic_from_json
from .clean_text import clean_text
from .torch_tools import torch_parameters,torch_get_input_shape

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
    "torch_get_input_shape",
    "clean_text"
]
