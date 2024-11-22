import sys,os
sys.stdout = open(os.devnull, 'w')
from .fast_llm import FastLLM
from .fast_vision import FastVLLM
sys.stdout = sys.__stdout__