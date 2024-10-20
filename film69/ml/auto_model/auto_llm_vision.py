from transformers import TextIteratorStreamer, AutoProcessor, AutoModelForVision2Seq
from threading import Thread
from .format import convert_chat_template
from PIL import Image
import requests,torch

class AutoModelLlmVision:
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
        self.model =  AutoModelForVision2Seq.from_pretrained(self.model_name,**parametor_model)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=False, skip_special_tokens=True)
        self.chat_template= self.processor.chat_template
        self.history=[]
        self.images=[]
        print("Model and processor loaded successfully")
    
    def generate(self,text:str,
            image =None,
            max_new_tokens:int=512,
            stream:bool=False,
            do_sample:bool=True,
            temperature:float=0.4,
            top_p:float=0.9,
            history_save:bool=True,
            custom_chat_template:bool=False
        ):
        if custom_chat_template:self.processor.chat_template=convert_chat_template(self.custom_chat_template,self.processor.chat_template)
        else:self.processor.chat_template=self.chat_template
        
        if image==None:
            me={"role": "user", "content": [{"type": "text", "text": text}]}
        else:
            me={"role": "user", "content": [{"type": "image"},{"type": "text", "text": text}]}
            
        if history_save:self.history.append(me)
        
        self.images.append(image)
        
        input_text = self.processor.apply_chat_template(self.history if history_save else [me], add_generation_prompt=True)
        inputs = self.processor(self.images, input_text, return_tensors="pt").to(self.model.device)
        terminators = [self.processor.tokenizer.eos_token_id]
        
        if not history_save:
            del self.images[-1]
        
        if stream==True:
            thread = Thread(target=self.model.generate, kwargs=
                            {
                            "streamer": self.streamer,
                            "max_new_tokens": max_new_tokens,
                            "eos_token_id":terminators,
                            "do_sample":do_sample,
                            "temperature":temperature,
                            "top_p":top_p,
                            **inputs
                            })
            thread.start()
            
            def inner():
                i=0
                text_out=""
                if history_save:self.history.append({"role": "user", "content": [{"type": "text", "text": text_out}]})
                for new_text in self.streamer:
                    i+=1
                    if i!=1:
                        text_out+=new_text
                        if history_save:self.history[-1]={"role": "user", "content": [{"type": "text", "text": text_out}]}
                        for k in new_text:
                            yield  k

                thread.join() 
            return inner()
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.4,
                top_p=0.95,
            )
            text_out=self.processor.decode(outputs[0][16:-1])

            if history_save:self.history.append({"role": "user", "content": [{"type": "text", "text": text_out}]})
            return text_out
        
if __name__ == "__main__":
    model=AutoModelLlmVision()
    model.load_model("Llama-3.2-11B-Vision-Instruct",torch_dtype=torch.bfloat16,device_map="auto",)
    
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    for i in model.generate("ในรูปคืออะไร",image,stream=True,history_save=False,max_new_tokens=100):
        print(i,end="")
    model.generate("ในรูปคืออะไร",image,history_save=False,max_new_tokens=30)