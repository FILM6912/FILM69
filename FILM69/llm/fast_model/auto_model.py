from .fast_llm import FastLLM
from .fast_vision import FastVLLM

class FastAutoModel:
    def __new__(cls, model_name: str, dtype: str = None, load_in_4bit: bool = False, **kwargs):
        try:
            base_class = FastLLM()
            base_class.load_model(model_name, dtype=dtype, load_in_4bit=load_in_4bit, **kwargs)
        except:
            base_class = FastVLLM()
            base_class.load_model(model_name, dtype=dtype, load_in_4bit=load_in_4bit, **kwargs)
        return base_class
