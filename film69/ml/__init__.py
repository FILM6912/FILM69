from .whisper import Whisper
from .whisper import eval as whisper_eval
from .model import LLMModel
from .vectordb import VectorDB
try:from .fast_model import FastLLM
except:pass
try:from .llama import Llama
except:pass

