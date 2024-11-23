from .fast_llm import FastLLM
from .fast_vision import FastVLLM
class AutoModel:
    def __init__(self,) -> None:
        self.model=None

    def load_model_(self,model_name,dtype=None,load_in_4bit=False,**kwargs)->FastLLM | FastVLLM:
        try:
            self=FastLLM()
            self.load_model(model_name,dtype=None,load_in_4bit=False,**kwargs)
        except:
            self=FastVLLM()
            self.load_model(model_name,dtype=None,load_in_4bit=False,**kwargs)
            
