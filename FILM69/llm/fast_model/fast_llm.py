import shutil
import os
import sys
from unsloth import FastLanguageModel
import datasets
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments, TextIteratorStreamer
import torch
from threading import Thread
from unsloth import is_bfloat16_supported
from pathlib import Path


class FastLLM:
    def __init__(self):
        self.chat_history = []
        self.quantization_method = {
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

        self.chat_template = {
            "Llama3": "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
            "Alpaca": "\n\n### Instruction:\n{}\n\n### Response:\n{}\n\n",
        }
        self.chat_template_model = {
            "Llama3": {
                "before_system": "<|start_header_id|>system<|end_header_id|>\n\n",
                "after_system": "<|eot_id|>",
                "before_user": "<|start_header_id|>user<|end_header_id|>\n\n",
                "after_user": "<|eot_id|>",
                "before_assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "after_assistant": "<|eot_id|>",
            },
            "Alpaca": {
                "before_system": "",
                "after_system": "\n\n",
                "before_user": "### Instruction:\n",
                "after_user": "\n\n",
                "before_assistant": "### Response:\n",
                "after_assistant": "\n\n",
            },
        }
        self.chat_format = "model"

    def apply_chat_template(self, message):
        """Applies the selected chat template to the message."""
        if self.chat_format not in self.chat_template_model:
            raise ValueError(f"Chat template {self.chat_format} not found.")

        template = self.chat_template_model[self.chat_format]
        message_format = ""
        for item in message:
            role = item["role"]
            content = item["content"]
            if role == "system":
                message_format += template["before_system"] + content + template["after_system"]
            elif role == "user":
                message_format += template["before_user"] + content + template["after_user"]
            elif role == "assistant":
                message_format += template["before_assistant"] + content + template["after_assistant"]

        if message[-1]["role"] != "assistant":
            message_format += template["before_assistant"]
        return message_format

    def load_model(self, model_name, dtype=None, load_in_4bit=False, **kwargs):
        """Loads the specified model."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, dtype=dtype, load_in_4bit=load_in_4bit, **kwargs
        )

        messages = [
            {"role": "system", "content": "{}"},
            {"role": "user", "content": "{}"},
            {"role": "assistant", "content": "{}"},
        ]
        formatted_chat = self.tokenizer.apply_chat_template(messages, tokenize=False)
        self.chat_template["model"] = formatted_chat

    def load_dataset(self, df=None, chat_template="model", add_eot=True, additional_information=False):
        """Loads and formats the dataset."""
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        if additional_information:
            return None

        if chat_template in self.chat_template:
            chat_template = self.chat_template[chat_template]

        EOS_TOKEN = self.tokenizer.eos_token

        def formatting_prompts_func(data_in):
            data = [data_in[i] for i in list(data_in.keys())]
            texts = []
            for data_tuple in zip(*data):
                text = chat_template.format(*data_tuple)
                if add_eot:
                    text += EOS_TOKEN
                texts.append(text)
            return {"text": texts}

        dataset = datasets.Dataset.from_pandas(df)
        self.dataset = dataset.map(formatting_prompts_func, batched=True)
        return self.dataset

    def trainer(
        self,
        max_seq_length=1024,
        learning_rate=2e-4,
        output_dir="outputs",
        callbacks=None,
        dataset_num_proc=2,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        seed=3407,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        report_to="none",
        **kwargs,
    ):
        """Configures the SFTTrainer."""
        self._trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=False,
            callbacks=callbacks,
            args=TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim=optim,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                output_dir=output_dir,
                report_to=report_to,
                **kwargs,
            ),
        )
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    def start_train(self):
        """Starts the training process."""
        self._trainer.train()

    def save_model(self, model_name, save_method="merged_16bit", **kwargs):
        """Saves the trained model."""
        self.model.save_pretrained_merged(model_name, self.tokenizer, save_method=save_method, **kwargs)

    def generate(
        self,
        text: str,
        max_new_tokens: int = 512,
        stream: bool = False,
        history_save: bool = True,
        temperature=0.4,
        top_p=0.9,
        end: list[str] = None,
        apply_chat_template=True,
        **kwargs,
    ):
        """Generates text based on the input."""
        FastLanguageModel.for_inference(self.model)

        if end is None:
            end = [self.tokenizer.eos_token]

        if history_save:
            self.chat_history.append({"role": "user", "content": text})

        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=False, skip_special_tokens=True, do_sample=True, temperature=temperature, top_p=top_p
        )

        if apply_chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                self.chat_history if history_save else [{"role": "user", "content": text}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            self.chat_format = apply_chat_template
            input_ids = self.tokenizer(
                self.apply_chat_template(self.chat_history if history_save else [{"role": "user", "content": text}]),
                return_tensors="pt",
            ).to(self.model.device)

        terminators = [self.tokenizer.eos_token_id] + [self.tokenizer.convert_tokens_to_ids(i) for i in end]

        def generate_with_params(input_ids,**kwargs):
            return self.model.generate(input_ids,**kwargs)

        generate_params = {
            "streamer": self.streamer,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": terminators,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs,
        }

        if stream:
            thread = Thread(target=generate_with_params, kwargs={"input_ids":input_ids,**generate_params} if apply_chat_template else {**input_ids,**generate_params})
            thread.start()

            def inner():
                text_out = ""
                if history_save:
                    self.chat_history.append({"role": "assistant", "content": text_out})
                for new_text in self.streamer:
                    text_out += new_text
                    if history_save:
                        self.chat_history[-1] = {"role": "assistant", "content": text_out}
                    yield from new_text
                thread.join()

            return inner()
        else:
            outputs = generate_with_params(input_ids,**generate_params)
            response = outputs[0][input_ids.shape[-1] :]
            text_out = self.tokenizer.decode(response, skip_special_tokens=True)

            if history_save:
                self.chat_history.append({"role": "assistant", "content": text_out})
            return text_out

    def export_to_GGUF(
        self,
        model_name="model",
        quantization_method=["q4_k_m", "q8_0", "f16"],
        save_original_model=False,
        max_size_gguf="49G",
        build_gpu=False,
        save_original_gguf=False,
        **kwargs,
    ):
        """Exports the model to GGUF format."""
        FastLanguageModel.for_inference(self.model)
        self.model.save_pretrained_gguf(model_name, self.tokenizer, quantization_method=quantization_method, **kwargs)

        source_directory = Path(model_name)
        gguf_directory = source_directory / "GGUF"
        max_size_gguf = max_size_gguf.upper()

        gguf_directory.mkdir(exist_ok=True)
        for file_path in source_directory.rglob("*unsloth*"):
            if file_path.is_file():
                new_file_name = file_path.name.replace("unsloth", model_name)
                new_file_path = gguf_directory / str(new_file_name).split("/")[-1]
                shutil.move(str(file_path), str(new_file_path))
                print(f"saved {new_file_path}")

        if not save_original_model:
            for item in os.listdir(model_name):
                item_path = os.path.join(model_name, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

        folder_path = f"{model_name}/GGUF"
        files_path, files_size = self.__check_file__(folder_path)

        if max(files_size) > self._convert_to_gb(max_size_gguf):
            for i in files_path:
                new_path = os.path.join(folder_path, i.split(".")[-2])
                os.makedirs(new_path, exist_ok=True)
                shutil.move(i, os.path.join(new_path, i.split("/")[-1]))

        if os.system("./llama.cpp/llama-gguf-split") != 256:
            build_gpu_command = "-DGGML_CUDA=ON"
            command = f"""
                cd llama.cpp && \
                cmake -B build {build_gpu_command if build_gpu else ''} && \
                cmake --build build --config Release && \
                cp build/bin/llama-* .
                """
            os.system(command)

        files_path, files_size = self.__check_file__(folder_path)
        for i in range(len(files_path)):
            if files_size[i] > self._convert_to_gb(max_size_gguf):
                command = f"""./llama.cpp/llama-gguf-split --split \
                    --split-max-size {max_size_gguf}\
                    {files_path[i]} {files_path[i][:-5]}
                """
                os.system(command)
                if not save_original_gguf:
                    os.remove(files_path[i])

    def _convert_to_gb(self, size_str):
        """Converts a size string (e.g., '49G') to GB."""
        unit_multipliers = {"M": 1 / 1024, "G": 1}
        num = float(size_str[:-1])
        unit = size_str[-1]
        return num * unit_multipliers.get(unit, 1)

    def __check_file__(self, path):
        """Checks files in a directory and returns their paths and sizes."""
        files_path = []
        files_size = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                file_size_gb = file_size / (1024**3)
                files_path.append(file_path)
                files_size.append(file_size_gb)
        return files_path, files_size

    def export_GGUF_push_to_hub(self, model_name="model", quantization_method=["q4_k_m", "q8_0", "f16"], token="", **kwargs):
        """Exports the model to GGUF and pushes it to the Hugging Face Hub."""
        self.model.push_to_hub_gguf(
            model_name, self.tokenizer, quantization_method=quantization_method, token=token, **kwargs
        )

    def save_model_to_hub(self, repo_id, save_method="merged_16bit", **kwargs):
        """Saves the model to the Hugging Face Hub."""
        self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method=save_method, **kwargs)
