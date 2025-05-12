
import json
from operator import itemgetter
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
from uuid import uuid4

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    _convert_any_typed_dicts_to_pydantic as convert_any_typed_dicts_to_pydantic,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from ollama import AsyncClient, Client, Message, Options
from pydantic import BaseModel, PrivateAttr, model_validator
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import Self, is_typeddict
# from ..fast_model.auto_model import FastAutoModel
from fast_langchain_llm import FastLangchainLLM
from PIL import Image
import base64
from io import BytesIO


def _get_usage_metadata_from_generation_info(
    generation_info: Optional[Mapping[str, Any]],
) -> Optional[UsageMetadata]:
    """Get usage metadata from ollama generation info mapping."""
    if generation_info is None:
        return None
    input_tokens: Optional[int] = generation_info.get("prompt_eval_count")
    output_tokens: Optional[int] = generation_info.get("eval_count")
    if input_tokens is not None and output_tokens is not None:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    return None


def _parse_json_string(
    json_string: str, raw_tool_call: dict[str, Any], skip: bool
) -> Any:
    """Attempt to parse a JSON string for tool calling.

    Args:
        json_string: JSON string to parse.
        skip: Whether to ignore parsing errors and return the value anyways.
        raw_tool_call: Raw tool call to include in error message.

    Returns:
        The parsed JSON string.

    Raises:
        OutputParserException: If the JSON string wrong invalid and skip=False.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        if skip:
            return json_string
        msg = (
            f"Function {raw_tool_call['function']['name']} arguments:\n\n"
            f"{raw_tool_call['function']['arguments']}\n\nare not valid JSON. "
            f"Received JSONDecodeError {e}"
        )
        raise OutputParserException(msg) from e
    except TypeError as e:
        if skip:
            return json_string
        msg = (
            f"Function {raw_tool_call['function']['name']} arguments:\n\n"
            f"{raw_tool_call['function']['arguments']}\n\nare not a string or a "
            f"dictionary. Received TypeError {e}"
        )
        raise OutputParserException(msg) from e


def _parse_arguments_from_tool_call(
    raw_tool_call: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Parse arguments by trying to parse any shallowly nested string-encoded JSON.

    Band-aid fix for issue in Ollama with inconsistent tool call argument structure.
    Should be removed/changed if fixed upstream.
    See https://github.com/ollama/ollama/issues/6155
    """
    if "function" not in raw_tool_call:
        return None
    arguments = raw_tool_call["function"]["arguments"]
    parsed_arguments = {}
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if isinstance(value, str):
                parsed_arguments[key] = _parse_json_string(
                    value, skip=True, raw_tool_call=raw_tool_call
                )
            else:
                parsed_arguments[key] = value
    else:
        parsed_arguments = _parse_json_string(
            arguments, skip=False, raw_tool_call=raw_tool_call
        )
    return parsed_arguments


def _get_tool_calls_from_response(
    response: Mapping[str, Any],
) -> List[ToolCall]:
    """Get tool calls from ollama response."""
    tool_calls = []
    if "message" in response:
        if raw_tool_calls := response["message"].get("tool_calls"):
            for tc in raw_tool_calls:
                tool_calls.append(
                    tool_call(
                        id=str(uuid4()),
                        name=tc["function"]["name"],
                        args=_parse_arguments_from_tool_call(tc) or {},
                    )
                )
    return tool_calls


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": tool_call["args"],
        },
    }


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class LangChainFastChat(BaseChatModel):
    model_name: str
    model_llm:FastLangchainLLM=None
    format_message:list=[]
    images:list=[]
    model_type:Literal["text","image"]="text"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del kwargs["model_name"]
        kwargs_model=kwargs
        self.model_llm=FastLangchainLLM(model_name=self.model_name,**kwargs_model)
    
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
        text=messages[-1].content
        print(text)
        tokens = self.model_llm.invoke(text,max_new_tokens=max_new_tokens,max_images_size=max_images_size)
        
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
        text=messages[-1].content
        
        tokens = self.model_llm.stream(text,max_new_tokens=max_new_tokens,max_images_size=max_images_size)
    
        ct_input_tokens = sum(len(message.content) for message in messages)
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
        
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: If provided, which tool for model to call. **This parameter
                is currently ignored as it is not supported by Ollama.**
            kwargs: Any additional parameters are passed directly to
                ``self.bind(**kwargs)``.
        """  # noqa: E501
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

