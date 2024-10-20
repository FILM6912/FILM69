from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
from threading import Thread
from .format import convert_chat_template

class AutoModelLlm:
    def __init__(self):
        self.custom_chat_template={
    "before_system": "<|start_header_id|>system<|end_header_id|>\n\n",
    "after_system": "<|eot_id|>",
    "before_user": "<|start_header_id|>user<|end_header_id|>\n\n",
    "after_user": "<|eot_id|>",
    "before_assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "after_assistant": "<|eot_id|>"
}
        
    def load_model(self,model_name,**parametor_model):
        self.model_name=model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,**parametor_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True)
        self.chat_template= self.tokenizer.chat_template
        self.history=[]
        print("Model and tokenizer loaded successfully")
    
    def generate(self,text:str,
            max_new_tokens:int=512,
            stream:bool=False,
            do_sample:bool=True,
            temperature:float=0.4,
            top_p:float=0.9,
            history_save:bool=True,
            custom_chat_template:bool=False
        ):
        if custom_chat_template:self.tokenizer.chat_template=convert_chat_template(self.custom_chat_template,self.tokenizer.chat_template)
        else:custom_chat_template:self.tokenizer.chat_template=self.chat_template
        
        if history_save:self.history.append({"role":"user","content":text})
        else: new_message= [{"role": "user","content": text}]
        
        input_ids = self.tokenizer.apply_chat_template(self.history if history_save else new_message,add_generation_prompt=True,return_tensors="pt").to(self.model.device)
        terminators = [self.tokenizer.eos_token_id]
        if stream==True:
            thread = Thread(target=self.model.generate, kwargs=
                            {"input_ids": input_ids,
                            "streamer": self.streamer,
                            "max_new_tokens": max_new_tokens,
                            "eos_token_id":terminators,
                            "do_sample":do_sample,
                            "temperature":temperature,
                            "top_p":top_p,
                                
                            })
            thread.start()
            
            def inner():
                i=0
                text_out=""
                if history_save:self.history.append({"role": "system","content": text_out})
                for new_text in self.streamer:
                    i+=1
                    if i!=1:
                        text_out+=new_text
                        if history_save:self.history[-1]={"role": "system","content": text_out}
                        for k in new_text:
                            yield  k

                thread.join() 
            return inner()
        else:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            response = outputs[0][input_ids.shape[-1]:]
            text_out=self.tokenizer.decode(response, skip_special_tokens=True)

            if history_save:self.history.append({"role": "system","content": text_out})
            return text_out
        
if __name__ == "__main__":
    model=AutoModelLlm()
    model.load_model("")
    for i in model.generate("คุณคือ",stream=True,max_new_tokens=100):
        print(i,end="")
    print(model.history)
    print("\n*"*50)
    print(model.history)