from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from ..fast_model import FastLLM

class LangChainFastLLM(LLM):
    # model_name:str
    model_llm: FastLLM=None
    # top_k:int=15
    # top_p:float=0.95
    # temperature:float=0.2
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_llm=FastLLM()
        self.model_llm.load_model(
            **kwargs
            )
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_new_tokens=512,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        
        return self.model_llm.generate(prompt,history_save=False,max_new_tokens=max_new_tokens)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_new_tokens=512,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for char in self.model_llm.generate(prompt,history_save=False,max_new_tokens=max_new_tokens,stream=True):
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "LangChainFastLLM",
        }

    @property
    def _llm_type(self) -> str:
        return "lang_chain_fast_llm"