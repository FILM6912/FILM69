import sys,os
from ...DisPrint import dis_print
with dis_print():
    from .fast_llm import FastLLM
    from .fast_vision import FastVLLM
    from .auto_model import FastAutoModel