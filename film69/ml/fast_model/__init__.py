
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
import shutil,os

class FastLLM:
    def __init__(self):
        self.history = []
        self.quantization_method = \
{
    "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized"      : "Recommended. Slow conversion. Fast inference, small files.",
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    "iq2_xxs" : "2.06 bpw quantization",
    "iq2_xs"  : "2.31 bpw quantization",
    "iq3_xxs" : "3.06 bpw quantization",
    "q3_k_xs" : "3-bit extra small quantization",
}

    def load_model(self,model_name,dtype=None,load_in_4bit=True,**kwargs):  
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            **kwargs
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

    def load_dataset(self,df,prompt_format = """\n\n### Instruction:\n{}\n\n### Response:\n{}\n\n"""):

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

        EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
        def formatting_prompts_func(data_in):
            data = [data_in[i] for i in list(data_in.keys())]
            texts = []
            for data_tuple in zip(*data):
                text = prompt_format.format(*data_tuple) + EOS_TOKEN
                texts.append(text)
            return { "text" : texts, }

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
        
        
    def generate(self,text:str,max_new_tokens:int=512,stream:bool=False,history_save:bool=True):
        FastLanguageModel.for_inference(self.model)
        if history_save:self.history.append({"role":"user","content":text})
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True,do_sample=True,temperature=0.4,top_p=0.9)
        input_ids = self.tokenizer.apply_chat_template(self.history if history_save else [{"role": "user","content": text}],add_generation_prompt=True,return_tensors="pt").to(self.model.device)
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
                if history_save:self.history.append({"role": "system","content": text_out})
                for new_text in self.streamer:
                    i+=1
                    if i!=1:
                        text_out+=new_text
                        if history_save:self.history[-1]={"role": "system","content": text_out}
                        yield  new_text

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
        
    def export_to_GGUF(self,model_name="model",quantization_method= ["q3_k_l","q4_k_m","q5_k_m","q8_0","f16"],save_original_model=False):
        FastLanguageModel.for_inference(self.model)
        self.model.save_pretrained_gguf(model_name, self.tokenizer, quantization_method = quantization_method)
        source_directory = Path(model_name)
        gguf_directory = source_directory / 'GGUF'

        gguf_directory.mkdir(exist_ok=True)
        for file_path in source_directory.rglob('*unsloth*'):
            if file_path.is_file():
                new_file_name = file_path.name.replace('unsloth', model_name)
                new_file_path = gguf_directory / new_file_name
                shutil.move(str(file_path), str(new_file_path))
                print(f'saved {new_file_path}')

        if not save_original_model:
            for item in os.listdir(model_name):
                item_path = os.path.join(model_name, item)
                if os.path.isfile(item_path):os.remove(item_path)

    def export_GGUF_push_to_hub(self,model_name="model",quantization_method= ["q3_k_l","q4_k_m","q5_k_m","q8_0","f16"],token=""):
        self.model.push_to_hub_gguf(
        model_name, 
        self.tokenizer,
        quantization_method = quantization_method,
        token = token,
    )
