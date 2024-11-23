from .fast_llm import FastLLM
from .fast_vision import FastVLLM
class AutoModel:
    def __init__(self,) -> None:
        self.model=None

    def load_model(self,model_name,dtype=None,load_in_4bit=False,**kwargs)->FastLLM | FastVLLM:
        try:
            model=FastLLM()
            model.load_model(model_name,dtype=None,load_in_4bit=False,**kwargs)
        except:
            model=FastVLLM()
            model.load_model(model_name,dtype=None,load_in_4bit=False,**kwargs)
            
        return model
