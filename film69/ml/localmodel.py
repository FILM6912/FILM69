from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
import torch
from threading import Thread

class LocalModel:
    def __init__(self, model_name):
        # model_name="d:\Model_LLM\llama-3-typhoon-v1.5x-8b-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True)
        self.history=[{"role":"user","content":"คุณคือผู้ช่วยชื่อ เสี่ยวซี่(XiaoXi) เป็นผู้หญิงและให้ตอบว่าคะ"},]
        print("Model and tokenizer loaded successfully")

    def generate(self,text,max_new_tokens=512,stream=False):
        self.history.append({"role":"user","content":text})
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
                self.history.append({"role": "system","content": text_out})

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

            self.history.append({"role": "system","content": text_out})
            return text_out
        
if __name__ == "__main__":
    model=LocalModel("d:\Model_LLM\\typhoon-7b-instruct-02-19-2024")
    for i in model.generate("คุณคือ",stream=True,max_new_tokens=100):
        print(i,end="")
    print(model.history)
    print(model.generate("คุณทำอะไรได้บ้าง",max_new_tokens=100))
    print(model.history)