import re
def convert_chat_template(format_dict,chat_template=None):
    """
from transformers import AutoProcessor
model_id = "Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
llama3_format = {
    "before_system": "<|start_header_id|>system<|end_header_id|>\n\n",
    "after_system": "<|eot_id|>",
    "before_user": "<|start_header_id|>user<|end_header_id|>\n\n",
    "after_user": "<|eot_id|>",
    "before_assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "after_assistant": "<|eot_id|>"
}
chat_template = processor.tokenizer.chat_template
converted_llama3 = convert_chat_template(llama3_format,chat_template)
print("Converted to Llama3:\n", converted_llama3)
    """
    # Update system messages
    chat_template = re.sub(
        r"{{- bos_token }}.*?<\|end_header_id\|>", 
        format_dict['before_system'] + format_dict['after_system'],
        chat_template, 
        flags=re.DOTALL
    )
    
    # Update user messages
    chat_template = re.sub(
        r"<\|start_header_id\|>user<\|end_header_id\|>\n\n.*?<\|eot_id\|>",
        format_dict['before_user'] + format_dict['after_user'],
        chat_template, 
        flags=re.DOTALL
    )
    
    # Update assistant messages
    chat_template = re.sub(
        r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n.*?<\|eot_id\|>",
        format_dict['before_assistant'] + format_dict['after_assistant'],
        chat_template,
        flags=re.DOTALL
    )
    
    return chat_template