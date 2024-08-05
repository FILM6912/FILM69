from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
import torch
from threading import Thread
from openai import OpenAI
class LLMModel:
    def __init__(self, 
                 model_name:str="d:\Model_LLM\llama-3-typhoon-v1.5x-8b-instruct",
                 local:bool=True,api=OpenAI(
                        api_key="your_api_key",
                        base_url="https://api.opentyphoon.ai/v1",),
                 **parametor_model
                        ):
        self.local=local
        self.model_name=model_name
        if local:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,**parametor_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True)
        else:self.api=api
        self.history=[{"role":"user","content":"คุณคือผู้ช่วยชื่อ เสี่ยวซี่(XiaoXi) เป็นผู้หญิงและให้ตอบว่าคะ"},]
        print("Model and tokenizer loaded successfully")

    def generate(self,text:str,max_new_tokens:int=512,stream:bool=False,history_save:bool=True):
        if self.local:return self.generate_locals(text,max_new_tokens,stream,history_save)
        else:return self.generate_api(text,max_new_tokens,stream,history_save)

    def generate_api(self,text:str,max_new_tokens:int=512,stream:bool=False,history_save:bool=True):
        self.history.append({"role":"user","content":text})
        text_out=""
        if stream:
            def inner():
                response= self.api.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user","content": text}],
                max_tokens=max_new_tokens,
                stream=True,)
                text_out=""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None: 
                        text_out+=chunk.choices[0].delta.content
                        yield chunk.choices[0].delta.content
                if history_save:self.history.append({"role": "system","content": text_out})
                else:self.history.pop()
            return inner()
            
        else:
            response= self.api.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user","content": text}],
                max_tokens=max_new_tokens,
            )
            text_out=response.choices[0].message.content
            if history_save:self.history.append({"role": "system","content": text_out})
            else:self.history.pop()
        return text_out

    def generate_locals(self,text:str,max_new_tokens:int=512,stream:bool=False,history_save:bool=True):
        if history_save:self.history.append({"role":"user","content":text})
        input_ids = self.tokenizer.apply_chat_template(self.history,add_generation_prompt=True,return_tensors="pt").to(self.model.device)
        terminators = [self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        if stream==True:
            thread = Thread(target=self.model.generate, kwargs=
                            {"input_ids": input_ids,
                            "streamer": self.streamer,
                            "max_new_tokens": max_new_tokens,
                            "eos_token_id":terminators,
                            "do_sample":True,
                            "temperature":0.4,
                            "top_p":0.9,
                                
                            })
            thread.start()
            
            def inner():
                i=0
                text_out=""
                for new_text in self.streamer:
                    i+=1
                    if i!=1:
                        text_out+=new_text
                        yield  new_text
                if history_save:self.history.append({"role": "system","content": text_out})
                else:self.history.pop()

                thread.join() 
            return inner()
        else:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.4,
                top_p=0.95,
            )
            response = outputs[0][input_ids.shape[-1]:]
            text_out=self.tokenizer.decode(response, skip_special_tokens=True)

            if history_save:self.history.append({"role": "system","content": text_out})
            else:self.history.pop()
            return text_out
        
if __name__ == "__main__":
    api=OpenAI(api_key="your_api_key",base_url="https://api.opentyphoon.ai/v1",)
    model=LLMModel(api=api,model_name="typhoon-v1.5x-70b-instruct",local=False)
    # for i in model.generate("คุณคือ",stream=True,max_new_tokens=100):
    #     print(i,end="")
    # print(model.history)
    # print("\n*"*50)
    print(model.generate_api("รู้จักประเทศไทยไหม",max_new_tokens=100))
    # print(model.history)