from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
import torch
from threading import Thread
from openai import OpenAI
from unsloth import FastLanguageModel
import datasets
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
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
        # self.history=[{"role":"user","content":"คุณคือผู้ช่วยชื่อ เสี่ยวซี่(XiaoXi) เป็นผู้หญิงและให้ตอบว่าคะ"},]
        self.history=[]
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
        

class LLMModelTrain:
    def load_model(self,model_name,dtype=None,load_in_4bit=True,**kwargs):  
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            **kwargs
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

    def load_dataset(self,df):
        alpaca_prompt = """
        ### Instruction:
        {}

        ### Response:
        {}"""

        EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            outputs      = examples["output"]
            texts = []
            for instruction, output in zip(instructions, outputs):
                text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
                texts.append(text)
            return { "text" : texts, }
        pass

        dataset = datasets.Dataset.from_pandas(df)
        self.dataset=dataset.map(formatting_prompts_func, batched = True,)
        return self.dataset

    def trainer(self,max_seq_length=1024,max_step=60,learning_rate=2e-4,output_dir = "outputs",**kwargs):
        self._trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = max_step,
                learning_rate = learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = output_dir,
                **kwargs
            ),
        )
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        return self._trainer

    def start_train(self):
        return self._trainer.train()
    
    def save_model(self,model_name,save_method = "merged_16bit",**kwargs):
        self.model.save_pretrained_merged(model_name, self.tokenizer, save_method = save_method,**kwargs)
if __name__ == "__main__":
    api=OpenAI(api_key="your_api_key",base_url="https://api.opentyphoon.ai/v1",)
    model=LLMModel(api=api,model_name="typhoon-v1.5x-70b-instruct",local=False)
    # for i in model.generate("คุณคือ",stream=True,max_new_tokens=100):
    #     print(i,end="")
    # print(model.history)
    # print("\n*"*50)
    print(model.generate_api("รู้จักประเทศไทยไหม",max_new_tokens=100))
    # print(model.history)