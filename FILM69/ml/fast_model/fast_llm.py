import shutil,os,sys
from unsloth import FastLanguageModel
import datasets
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments,TextIteratorStreamer
# from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
import torch
from threading import Thread
from unsloth import is_bfloat16_supported
from pathlib import Path

class FastLLM:
    def __init__(self):
        self.history = []
        self.quantization_method ={
    "not_quantized": "แนะนำ คอนเวอร์ชันรวดเร็ว แต่การอนุมานช้า ไฟล์ขนาดใหญ่",
    "fast_quantized": "แนะนำ คอนเวอร์ชันรวดเร็ว การอนุมานโอเค ขนาดไฟล์โอเค",
    "quantized": "แนะนำ คอนเวอร์ชันช้า การอนุมานรวดเร็ว ไฟล์ขนาดเล็ก",
    "f32": "ไม่แนะนำ รักษาความแม่นยำ 100% แต่ช้าและใช้หน่วยความจำมาก",
    "f16": "คอนเวอร์ชันเร็วที่สุด รักษาความแม่นยำ 100% แต่ช้าและใช้หน่วยความจำมาก",
    "q8_0": "คอนเวอร์ชันเร็ว ใช้ทรัพยากรสูง แต่โดยทั่วไปถือว่ารับได้",
    "q4_k_m": "แนะนำ ใช้ Q6_K สำหรับครึ่งหนึ่งของ attention.wv และ feed_forward.w2 tensors ส่วนที่เหลือเป็น Q4_K",
    "q5_k_m": "แนะนำ ใช้ Q6_K สำหรับครึ่งหนึ่งของ attention.wv และ feed_forward.w2 tensors ส่วนที่เหลือเป็น Q5_K",
    "q2_k": "ใช้ Q4_K สำหรับ attention.vw และ feed_forward.w2 tensors และ Q2_K สำหรับ tensors อื่นๆ",
    "q3_k_l": "ใช้ Q5_K สำหรับ attention.wv, attention.wo และ feed_forward.w2 tensors ส่วนที่เหลือเป็น Q3_K",
    "q3_k_m": "ใช้ Q4_K สำหรับ attention.wv, attention.wo และ feed_forward.w2 tensors ส่วนที่เหลือเป็น Q3_K",
    "q3_k_s": "ใช้ Q3_K สำหรับทุก tensor",
    "q4_0": "วิธีการควอนตัมแบบดั้งเดิม 4-bit",
    "q4_1": "มีความแม่นยำสูงกว่าค่า q4_0 แต่ไม่สูงเท่าค่า q5_0 อย่างไรก็ตามมีการอนุมานที่เร็วกว่าโมเดล q5",
    "q4_k_s": "ใช้ Q4_K สำหรับทุก tensor",
    "q4_k": "ชื่อเล่นสำหรับ q4_k_m",
    "q5_k": "ชื่อเล่นสำหรับ q5_k_m",
    "q5_0": "ความแม่นยำสูงขึ้น ใช้ทรัพยากรมากขึ้นและการอนุมานช้าลง",
    "q5_1": "ความแม่นยำสูงขึ้นอีก ใช้ทรัพยากรมากขึ้นและการอนุมานช้าลง",
    "q5_k_s": "ใช้ Q5_K สำหรับทุก tensor",
    "q6_k": "ใช้ Q8_K สำหรับทุก tensor",
    "iq2_xxs": "ควอนตัม 2.06 bpw",
    "iq2_xs": "ควอนตัม 2.31 bpw",
    "iq3_xxs": "ควอนตัม 3.06 bpw",
    "q3_k_xs": "ควอนตัมขนาดเล็กพิเศษ 3-bit",
}

        self.chat_template={
            "Llama3":"<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
            "Alpaca":"\n\n### Instruction:\n{}\n\n### Response:\n{}\n\n" 
        }
        self.chat_template_model={
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

    def apply_chat_template(self,message):
        
        if self.chat_format in self.chat_template_model.keys():
            before_system=self.chat_template_model[self.chat_format]["before_system"]
            after_system=self.chat_template_model[self.chat_format]["after_system"]
            before_user=self.chat_template_model[self.chat_format]["before_user"]
            after_user=self.chat_template_model[self.chat_format]["after_user"]
            before_assistant=self.chat_template_model[self.chat_format]["before_assistant"]
            after_assistant=self.chat_template_model[self.chat_format]["after_assistant"]
            message_format=""
            for i in message:
                if i["role"]=="system":message_format+=before_system+i["content"]+after_system
                elif i["role"]=="user":message_format+=before_user+i["content"]+after_user
                elif i["role"]=="assistant":message_format+=before_assistant+i["content"]+after_assistant
            if message[-1]["role"]!="assistant":message_format+=before_assistant
            return message_format
            
        else:return ValueError(f"Chat template {self.chat_format} not found.")
    
    def load_model(self,model_name,dtype=None,load_in_4bit=False,**kwargs): 
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            **kwargs
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
    

    def load_dataset(self,df=None,chat_template = None,add_eot=True,additional_information=False):
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

        if additional_information==False:
            if chat_template in self.chat_template.keys():
                chat_template = self.chat_template [chat_template]
            
            EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
            def formatting_prompts_func(data_in):
                data = [data_in[i] for i in list(data_in.keys())]
                texts = []
                for data_tuple in zip(*data):
                    if add_eot:text = chat_template.format(*data_tuple) + EOS_TOKEN
                    else:text = chat_template.format(*data_tuple)
                    texts.append(text)
                return { "text" : texts, }

            dataset = datasets.Dataset.from_pandas(df)
            self.dataset=dataset.map(formatting_prompts_func, batched = True,)
            return self.dataset
        return None

    def trainer(self,max_seq_length=1024,learning_rate=2e-4,output_dir = "outputs",callbacks=None,**kwargs):
        "trainer(self,max_seq_length=1024,max_step=60 or num_train_epochs=3,learning_rate=2e-4,output_dir = 'outputs',callbacks=None)"
        self._trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            callbacks=callbacks,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.

                learning_rate = learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
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

    def start_train(self):
        self._trainer.train()

    def save_model(self,model_name,save_method = "merged_16bit",**kwargs):
        self.model.save_pretrained_merged(model_name, self.tokenizer, save_method = save_method,**kwargs)

    def generate(self,
        text:str,
        max_new_tokens:int=512,
        stream:bool=False,
        history_save:bool=True,
        temperature=0.4,
        top_p=0.9,
        end:list[str]=None,
        apply_chat_template=True,
        **kwargs):
        FastLanguageModel.for_inference(self.model)
        
        if end==None:
            end=[self.tokenizer.eos_token]
        
        if history_save:self.history.append({"role":"user","content":text})
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True,do_sample=True,temperature=0.4,top_p=0.9)
        if apply_chat_template==True :
            input_ids = self.tokenizer.apply_chat_template(self.history if history_save else [{"role": "user","content": text}],add_generation_prompt=True,return_tensors="pt").to(self.model.device)
        else:
            self.chat_format=apply_chat_template
            input_ids=self.tokenizer(self.apply_chat_template(self.history if history_save else [{"role": "user","content": text}]), return_tensors = "pt").to(self.model.device)
        
        terminators = [self.tokenizer.eos_token_id]+[self.tokenizer.convert_tokens_to_ids(i) for i in end]
        if stream==True:
            thread = Thread(target=self.model.generate, kwargs=
                            {
                            "input_ids": input_ids,
                            "streamer": self.streamer,
                            "max_new_tokens": max_new_tokens,
                            "eos_token_id":terminators,
                            "do_sample":True,
                            "temperature":temperature,
                            "top_p":top_p,
                            **kwargs
                                
                            }) if apply_chat_template==True else Thread(target=self.model.generate, kwargs=
                            {
                            **input_ids,
                            "streamer": self.streamer,
                            "max_new_tokens": max_new_tokens,
                            "eos_token_id":terminators,
                            "do_sample":True,
                            "temperature":temperature,
                            "top_p":top_p,
                            **kwargs
                                
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
                        for te in new_text:yield  te

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
            return text_out
        
    def export_to_GGUF(self,model_name="model",quantization_method= ["q3_k_l","q4_k_m","q5_k_m","q8_0","f16"],save_original_model=False,**kwargs):
        FastLanguageModel.for_inference(self.model)
        self.model.save_pretrained_gguf(model_name, self.tokenizer, quantization_method = quantization_method,**kwargs)
        source_directory = Path(model_name)
        gguf_directory = source_directory / 'GGUF'

        gguf_directory.mkdir(exist_ok=True)
        for file_path in source_directory.rglob('*unsloth*'):
            if file_path.is_file():
                new_file_name = file_path.name.replace('unsloth', model_name)
                new_file_path = gguf_directory / str(new_file_name).split("/")[-1]
                shutil.move(str(file_path), str(new_file_path))
                print(f'saved {new_file_path}')

        if not save_original_model:
            for item in os.listdir(model_name):
                item_path = os.path.join(model_name, item)
                if os.path.isfile(item_path):os.remove(item_path)

    def export_GGUF_push_to_hub(self,model_name="model",quantization_method= ["q3_k_l","q4_k_m","q5_k_m","q8_0","f16"],token="",**kwargs):
        self.model.push_to_hub_gguf(
        model_name, 
        self.tokenizer,
        quantization_method = quantization_method,
        token = token,
        **kwargs
    )

    def save_model_to_hub(self,repo_id,save_method = "merged_16bit",**kwargs):
        self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method = save_method, **kwargs)
