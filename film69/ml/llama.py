from llama_cpp import Llama as Llama_cpp

class Llama():
    def __init__(self):
        self.history=[]
        self.chat_templet_model={
            "Llama3":{
                "before_system":"<|start_header_id|>system<|end_header_id|>\n\n",
                "after_system":"<|eot_id|>",
                "before_user":"<|start_header_id|>user<|end_header_id|>\n\n",
                "after_user":"<|eot_id|>",
                "before_assistant":"<|start_header_id|>assistant<|end_header_id|>\n\n",
                "after_assistant":"<|eot_id|>"
            },
            "Alpaca":{
                "before_system":"",
                "after_system":"\n\n",
                "before_user":"### Instruction:\n",
                "after_user":"\n\n",
                "before_assistant":"### Response:\n",
                "after_assistant":"\n\n" 
            }
        }
        self.chat_format="Llama3"

    def load_model(self,model_path,n_gpu_layers=-1,n_ctx=2048,verbose=False,**kwargs):
        self.llm = Llama_cpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            **kwargs
        )
    
    def chat_templet(self,message):
        
        if self.chat_format in self.chat_templet_model.keys():
            before_system=self.chat_templet_model[self.chat_format]["before_system"]
            after_system=self.chat_templet_model[self.chat_format]["after_system"]
            before_user=self.chat_templet_model[self.chat_format]["before_user"]
            after_user=self.chat_templet_model[self.chat_format]["after_user"]
            before_assistant=self.chat_templet_model[self.chat_format]["before_assistant"]
            after_assistant=self.chat_templet_model[self.chat_format]["after_assistant"]
            message_format=""
            for i in message:
                if i["role"]=="system":message_format+=before_system+i["content"]+after_system
                elif i["role"]=="user":message_format+=before_user+i["content"]+after_user
                elif i["role"]=="assistant":message_format+=before_assistant+i["content"]+after_assistant
            if message[-1]["role"]!="assistant":message_format+=before_assistant
            return message_format
            
        else:return ValueError(f"Chat template {self.chat_format} not found.")
    
    def generate(self,message:list[dict], stream=False,max_tokens=512,show_all=False,**kwargs):
        text=self.chat_templet(message)
        if stream == True:
            def inner():
                for output in self.llm(text, stream=True,max_tokens=max_tokens,**kwargs):
                    yield output["choices"][0]["text"] if not show_all else output
            return inner()
        else:
            out=self.llm(text,max_tokens=max_tokens,**kwargs)
            return out["choices"][0]["text"] if not show_all else out
        