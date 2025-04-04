import warnings
import os
import shutil
from pathlib import Path
import json
from threading import Thread

import torch
from PIL import Image
from datasets import load_dataset
from transformers import TextIteratorStreamer
from trl import SFTTrainer, SFTConfig

from unsloth import FastModel as _FastModel
from unsloth import UnslothVisionDataCollator, is_bf16_supported
from unsloth_zoo.vision_utils import process_vision_info, get_padding_tokens_ids, _get_dtype

warnings.simplefilter("ignore", SyntaxWarning)


class FastModel:
    """
    A class for loading, training, and generating text with large language models (LLMs),
    with support for vision-language models.
    """

    def __init__(self) -> None:
        """Initializes the FastModel with empty chat and image histories."""
        self.chat_history = []
        self.images_history = []
        self.model = None
        self.processor = None
        self.converted_dataset = None
        self._trainer = None
        self.load_in_4bit = False
        self.load_in_8bit = False
        self.streamer = None

    def load_model(self, model_name, dtype=None, load_in_4bit=False, load_in_8bit=False, **kwargs):
        """Loads a pre-trained model and processor.

        Args:
            model_name (str): The name or path of the pre-trained model.
            dtype (torch.dtype, optional): The data type to use for the model. Defaults to None.
            load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
            load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
            **kwargs: Additional keyword arguments to pass to `_FastModel.from_pretrained`.
        """
        self.model, self.processor = _FastModel.from_pretrained(
            model_name=model_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            **kwargs
        )
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

    def load_dataset(self, dataset):
        """Loads a dataset for training.

        Args:
            dataset: The dataset to load.
        """
        self.converted_dataset = dataset

    def save_model(self, model_name, save_method="merged_16bit", **kwargs):
        """Saves the model and processor.

        Args:
            model_name (str): The directory to save the model to.
            save_method (str, optional): The method to use for saving. Defaults to "merged_16bit".
            **kwargs: Additional keyword arguments to pass to `save_pretrained_merged` or `save_pretrained`.
        """
        if self.load_in_4bit or self.load_in_8bit:
            try:
                self.model.save_pretrained_merged(model_name, self.processor, save_method=save_method, **kwargs)
            except:
                self.load_dataset(None)
                self.model.save_pretrained_merged(model_name, self.processor, save_method=save_method, **kwargs)

            if save_method == "merged_16bit":
                config = json.loads(self.model.config.to_json_string())
                for key in ["_attn_implementation_autoset", "quantization_config"]:
                    try:
                        del config[key]
                    except KeyError:
                        pass
                with open(f"{model_name}/config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
        else:
            self.model.save_pretrained(model_name)
            self.processor.save_pretrained(model_name)

    def trainer(
        self,
        max_seq_length=2048,
        learning_rate=2e-4,
        output_dir="outputs",
        callbacks=None,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        dataset_num_proc=4,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        data_collator=None,
        **kwargs
    ):
        """Configures the SFTTrainer for fine-tuning.

        Args:
            max_seq_length (int, optional): Maximum sequence length. Defaults to 2048.
            learning_rate (float, optional): Learning rate. Defaults to 2e-4.
            output_dir (str, optional): Output directory for training artifacts. Defaults to "outputs".
            callbacks (list, optional): List of callbacks. Defaults to None.
            per_device_train_batch_size (int, optional): Batch size per device. Defaults to 2.
            gradient_accumulation_steps (int, optional): Gradient accumulation steps. Defaults to 4.
            warmup_steps (int, optional): Warmup steps. Defaults to 5.
            optim (str, optional): Optimizer. Defaults to "adamw_8bit".
            weight_decay (float, optional): Weight decay. Defaults to 0.01.
            lr_scheduler_type (str, optional): Learning rate scheduler type. Defaults to "linear".
            seed (int, optional): Random seed. Defaults to 3407.
            report_to (str, optional): Reporting integration. Defaults to "none".
            remove_unused_columns (bool, optional): Remove unused columns. Defaults to False.
            dataset_num_proc (int, optional): Number of processes for dataset loading. Defaults to 4.
            finetune_vision_layers (bool, optional): Fine-tune vision layers. Defaults to True.
            finetune_language_layers (bool, optional): Fine-tune language layers. Defaults to True.
            finetune_attention_modules (bool, optional): Fine-tune attention modules. Defaults to True.
            finetune_mlp_modules (bool, optional): Fine-tune MLP modules. Defaults to True.
            r (int, optional): LoRA rank. Defaults to 16.
            lora_alpha (int, optional): LoRA alpha. Defaults to 16.
            lora_dropout (float, optional): LoRA dropout. Defaults to 0.
            bias (str, optional): Bias type. Defaults to "none".
            random_state (int, optional): Random state. Defaults to 3407.
            use_rslora (bool, optional): Use rank-stabilized LoRA. Defaults to False.
            loftq_config (dict, optional): LoftQ configuration. Defaults to None.
            train_on_responses_only (bool, optional): Train on responses only. Defaults to False.
            instruction_part (str, optional): Instruction part template. Defaults to "<|start_header_id|>user<|end_header_id|>\n\n".
            response_part (str, optional): Response part template. Defaults to "<|start_header_id|>assistant<|end_header_id|>\n\n".
            **kwargs: Additional keyword arguments for SFTConfig.
        """
        self.model = _FastModel.get_peft_model(
            self.model,
            finetune_vision_layers=finetune_vision_layers,
            finetune_language_layers=finetune_language_layers,
            finetune_attention_modules=finetune_attention_modules,
            finetune_mlp_modules=finetune_mlp_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )
        
        if data_collator == None:
            data_collator = UnslothVisionDataCollator(
                self.model,
                self.processor,
                train_on_responses_only=False,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            )

        self._trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.processor,
            data_collator=data_collator,
            train_dataset=self.converted_dataset,
            callbacks=callbacks,
            args=SFTConfig(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                learning_rate=learning_rate,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                optim=optim,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                output_dir=output_dir,
                report_to=report_to,
                remove_unused_columns=remove_unused_columns,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=dataset_num_proc,
                max_seq_length=max_seq_length,
                **kwargs
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

    def resize_image_pil(self, image, max_size=1100):
        """Resizes a PIL image while maintaining aspect ratio.

        Args:
            image (PIL.Image): The input image.
            max_size (int, optional): The maximum size for the image's width or height. Defaults to 1100.

        Returns:
            PIL.Image: The resized image.
        """
        img_copy = image.copy()
        img_copy.thumbnail((max_size, max_size))
        return img_copy

    def generate(
        self,
        text: str = "",
        image: Image = None,
        max_new_tokens: int = 512,
        stream: bool = False,
        history_save: bool = True,
        temperature=0.4,
        top_p=0.9,
        max_images_size=1000,
        end: list[str] = None,
        **kwargs
    ):
        """Generates text based on a prompt and optional image.

        Args:
            text (str, optional): The text prompt. Defaults to "".
            image (PIL.Image, optional): An optional image. Defaults to None.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 512.
            stream (bool, optional): Whether to stream the output. Defaults to False.
            history_save (bool, optional): Whether to save the conversation history. Defaults to True.
            temperature (float, optional): Sampling temperature. Defaults to 0.4.
            top_p (float, optional): Top-p sampling. Defaults to 0.9.
            max_images_size (int, optional): Maximum size for image resizing. Defaults to 1000.
            end (list[str], optional): List of end tokens. Defaults to None.
            **kwargs: Additional keyword arguments for `model.generate`.

        Returns:
            str or generator: The generated text or a generator for streaming output.
        """
        _FastModel.for_inference(self.model)
        if end is None:
            end = [self.processor.tokenizer.eos_token]

        if image is None:
            messages = {"role": "user", "content": [{"type": "text", "text": text}]}
        else:
            image = self.resize_image_pil(image, max_images_size)
            self.images_history.append(image)
            messages = {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
            }

        self.chat_history.append(messages)

        self.streamer = TextIteratorStreamer(self.processor, skip_prompt=True, skip_special_tokens=True)
        terminators = [self.processor.tokenizer.eos_token_id] + [
            self.processor.tokenizer.convert_tokens_to_ids(i) for i in end
        ]

        input_ids = self.processor.apply_chat_template(
            self.chat_history if history_save else [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        check_image = self.chat_history[-1]["content"] if history_save else messages["content"]
        if not any(i["type"] == "image" for i in check_image):
            input_ids["input_ids"] = torch.tensor([[2] + input_ids["input_ids"].cpu().numpy().tolist()[0]]).to("cuda")
            input_ids["attention_mask"] = torch.tensor([[1] + input_ids["attention_mask"].cpu().numpy().tolist()[0]]).to(
                "cuda"
            )
            input_ids["token_type_ids"] = torch.tensor([[0] + input_ids["token_type_ids"].cpu().numpy().tolist()[0]]).to(
                "cuda"
            )

        if not history_save and text != "":
            del self.chat_history[-1]

        if stream:
            thread = Thread(
                target=self.model.generate,
                kwargs={
                    **input_ids,
                    "streamer": self.streamer,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "eos_token_id": terminators,
                    **kwargs,
                },
            )
            thread.start()

            def inner():
                i = 0
                text_out = ""
                for new_text in self.streamer:
                    i += 1
                    if i != 1:
                        text_out += new_text
                        for te in new_text:
                            yield te
                if history_save:
                    self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": text_out}]})

                thread.join()

            return inner()
        else:
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=terminators,
                **kwargs
            )

            response = outputs[0][input_ids["input_ids"].shape[-1]:]
            text_out = self.processor.decode(response, skip_special_tokens=True)
            if history_save:
                self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": text_out}]})

            return text_out

    def export_to_GGUF(
        self,
        model_name="model",
        quantization_method=["q4_k_m", "q8_0", "f16"],
        save_original_model=False,
        max_size_gguf="49G",
        build_gpu=False,
        save_original_gguf=False,
        **kwargs
    ):
        """Exports the model to GGUF format.

        Args:
            model_name (str, optional): The name of the model. Defaults to "model".
            quantization_method (list, optional): List of quantization methods. Defaults to ["q4_k_m", "q8_0", "f16"].
            save_original_model (bool, optional): Whether to save the original model. Defaults to False.
            max_size_gguf (str, optional): Maximum size for GGUF files (e.g., "49G"). Defaults to "49G".
            build_gpu (bool, optional): Whether to build llama.cpp with GPU support. Defaults to False.
            save_original_gguf (bool, optional): Whether to save the original GGUF file. Defaults to False.
            **kwargs: Additional keyword arguments for `save_pretrained_gguf`.
        """
        _FastModel.for_inference(self.model)
        self.model.save_pretrained_gguf(model_name, self.processor, quantization_type="Q8_0", **kwargs)
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
        """Converts a size string (e.g., "49G", "10M") to gigabytes.

        Args:
            size_str (str): The size string.

        Returns:
            float: The size in gigabytes.
        """
        unit_multipliers = {"M": 1 / 1024, "G": 1}

        num = float(size_str[:-1])
        unit = size_str[-1]

        return num * unit_multipliers.get(unit, 1)

    def __check_file__(self, path):
        """Checks files in a directory and returns their paths and sizes.

        Args:
            path (str): The directory path.

        Returns:
            tuple: A tuple containing lists of file paths and file sizes (in GB).
        """
        files_path = []
        files_size = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                file_size_gb = file_size / (1024 ** 3)
                files_path.append(file_path)
                files_size.append(file_size_gb)
        return files_path, files_size


if __name__ == "__main__":
    model = FastModel()
    model.load_model("unsloth/gemma-3-4b-it", load_in_4bit=True)
    dataset = load_dataset("unsloth/Radiology_mini", split="train")
    dataset = dataset.select(range(5))

    instruction = "You are an expert radiographer. Describe accurately what you see in this image."

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    # {"type" : "text",  "text"  : "สวัสดี"},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    # {"type" : "text",  "text"  : "สวัสดีครับคุณ film"} ]
                    {"type": "text", "text": sample["caption"]},
                ],
            },
        ]
        return {"messages": conversation}

    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    model.load_dataset(converted_dataset)
    model.trainer(max_steps=60, logging_steps=1)

    model.start_train()
