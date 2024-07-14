import os
from typing import Any, Optional
import argparse
from llama_index.llms.openai_like import OpenAILike

class Typhoon(OpenAILike):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://api.opentyphoon.ai/v1",
        is_chat_model: bool = True,
        is_function_calling_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("Typhoon_API_KEY", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Typhoon"

if __name__=="__main__":
    llm=Typhoon(model="typhoon-v1.5x-70b-instruct",api_key="you_api_key")
