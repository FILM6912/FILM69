from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from FILM69.llm import FastAutoModel
from PIL import Image
import base64
from io import BytesIO
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

class FastLangchainLLM(LLM):
    model_name: str
    model_llm:FastAutoModel=None
    format_message:list=[]
    images:list=[]
    model_type:Literal["text","image"]="text"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del kwargs["model_name"]
        kwargs_model=kwargs
        self.model_llm=FastAutoModel(model_name=self.model_name,**kwargs_model)
    
    def base64_to_pil_image(self,base64_str):
        base64_data = base64_str.split(",")[1] if "," in base64_str else base64_str
        image_data = base64.b64decode(base64_data)
        image_stream = BytesIO(image_data)
        pil_image = Image.open(image_stream)
        return pil_image
    
    def pil_image_to_base64(self,pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def apply_chat_template(self,message):
        self.format_message=[]
        self.images=[]
        for i in message:
            if type(i)==HumanMessage:
                _type="user"
            elif type(i) == AIMessage:_type="assistant"
            elif type(i)== SystemMessage:_type="system"
            
            content=[]
            if type(i.content) == list:
                for j in i.content:
                    if j["type"]=="text":content.append({'type': 'text', 'text': j['text']})
                    else:
                        content.append({'type': 'image'})
                        self.images.append(self.base64_to_pil_image(j["image_url"]))
            else:
                content.append({'type': 'text', 'text': i.content})

            self.format_message.append({'role': _type,'content':content})
            
    
    def _call(
       self,
        prompt:str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        max_new_tokens=8092,
        max_image_size=1000,
        image=None,
        **kwargs: Any,
    ) -> str:
      
        _kwargs={
            "history_save":False,
            "stream":False,
            "max_new_tokens":max_new_tokens,
        }
        if self.model_type=="image":
            _kwargs["max_images_size"]=max_image_size
            _kwargs["images"]= image
        
        tokens = self.model_llm.generate(prompt,**_kwargs)
        
        return tokens
    

    def _stream(
        self,
        prompt:str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        max_new_tokens=8092,
        max_image_size=1000,
        image=None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        _kwargs={
            "history_save":False,
            "stream":True,
            "max_new_tokens":max_new_tokens,
        }
        if self.model_type=="image":
            _kwargs["max_images_size"]=max_image_size
            _kwargs["images"]=image
        
        tokens = self.model_llm.generate(prompt,**_kwargs)
        
        for char in tokens:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        return "custom"