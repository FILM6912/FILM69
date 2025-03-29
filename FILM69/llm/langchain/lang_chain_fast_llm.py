from typing import Any, Dict, Iterator, List, Optional,Literal

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from ..fast_model.auto_model import FastAutoModel
from PIL import Image
import base64
from io import BytesIO
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

class LangChainFastLLM(BaseChatModel):
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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        max_new_tokens=8092,
        max_images_size=1000,
        **kwargs: Any,
    ) -> ChatResult:
        try:len(messages)
        except:messages=[messages]
        self.apply_chat_template(messages)
        
        self.model_llm.chat_history=self.format_message[:-1]
        
        
        _kwargs={
            "history_save":False,
            "stream":False,
            "max_new_tokens":max_new_tokens,
        }
        if self.model_type=="image":
            _kwargs["max_images_size"]=max_images_size
            _kwargs["images"]= self.images[-1] if self.images!=[] else None
        
        tokens = self.model_llm.generate([i["text"] for i in self.format_message[-1]["content"] if i["type"] == "text"][0],**_kwargs)
        
        ct_input_tokens = sum(len(message.content) for message in messages)
        ct_output_tokens = len(tokens)
        message = AIMessage(
            content=tokens,
            additional_kwargs={},
            response_metadata={
                "time_in_seconds": 3,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        max_new_tokens=8092,
        max_images_size=1000,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:len(messages)
        except:messages=[messages]
        ct_input_tokens = sum(len(message.content) for message in messages)
        
        self.apply_chat_template(messages)
        self.model_llm.chat_history=self.format_message[:-1]
        
        _kwargs={
            "history_save":False,
            "stream":True,
            "max_new_tokens":max_new_tokens,
        }
        if self.model_type=="image":
            _kwargs["max_images_size"]=max_images_size
            _kwargs["images"]=self.images[-1] if self.images!=[] else None
        
        tokens = self.model_llm.generate([i["text"] for i in self.format_message[-1]["content"] if i["type"] == "text"][0],**_kwargs)
        
        for token in tokens:
            usage_metadata = UsageMetadata(
                {
                    "input_tokens": ct_input_tokens,
                    "output_tokens": 1,
                    "total_tokens": ct_input_tokens + 1,
                }
            )
            ct_input_tokens = 0
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=token, usage_metadata=usage_metadata)
            )

            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
        }
