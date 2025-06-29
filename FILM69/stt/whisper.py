from unsloth import FastModel
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from unsloth import is_bfloat16_supported
from dataclasses import dataclass
from typing import Any, Dict, List, Union,Literal
from datasets import load_dataset, Audio
from transformers import pipeline
import evaluate
import torch
import warnings
warnings.simplefilter("ignore", FutureWarning)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    formatting_prompts_func: Any
    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        features=[self.formatting_prompts_func(example) for example in examples]
        
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch


class Whisper:
    def __init__(self) -> None:...
    
    
    def load_model(
        self,
        model_name_or_path,
        language = "Thai",
        config_language=None,
        task = "transcribe" or "translate",
        dtype=None,
        load_in_4bit=False,
        device_map="auto",
        **kwargs
        ):
        
        if device_map=="auto":self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:self.device = device_map
        self.base_model, self.tokenizer = FastModel.from_pretrained(
            model_name = model_name_or_path,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            auto_model = WhisperForConditionalGeneration,
            whisper_language = language,
            whisper_task = task,
            **kwargs
        )

        self.base_model.generation_config.language =f"<|{config_language}|>" if config_language!=None else None 
        self.base_model.generation_config.task = task
        self.base_model.config.suppress_tokens = []
        self.base_model.generation_config.forced_decoder_ids = None
    
    
    def formatting_prompts_func(self,example):
        audio_arrays = example['audio']['array']
        sampling_rate = example["audio"]["sampling_rate"]
        features = self.tokenizer.feature_extractor(
            audio_arrays, sampling_rate=sampling_rate
        )
        
        self.tokenizer.tokenizer.set_prefix_tokens(language=example["language"], task="transcribe") 
        
        tokenized_text = self.tokenizer.tokenizer(example["text"])
        return {
            "input_features": features.input_features[0],
            "labels": tokenized_text.input_ids,
        }
        
    def load_dataset(self,train_dataset,test_dataset=None,peft=True):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        if peft:
            self.model = FastModel.get_peft_model(
                self.base_model,
                r = 64,
                target_modules = ["q_proj", "v_proj"],
                lora_alpha = 64,
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 3407,
                use_rslora = False,
                loftq_config = None,
                task_type = None
            )
            
        else:
            self.model = self.base_model
    

    def compute_metrics(self,pred):
        self.metric = evaluate.load("wer")
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 *  self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def triner(
        self,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        remove_unused_columns=False,
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        eval_steps = None,
        eval_strategy:Literal["no","steps","epoch"]="no",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", 
        callbacks=None,
        
        **kwargs):
        """#num_train_epochs = 1
        max_steps = 1000"""
        
        self._trainer = Seq2SeqTrainer(
            model = self.model,
            train_dataset = self.train_dataset,
            data_collator = DataCollatorSpeechSeq2SeqWithPadding(
                processor=self.tokenizer,
                formatting_prompts_func=self.formatting_prompts_func
                ),
            eval_dataset = self.test_dataset if self.test_dataset else None,
            tokenizer = self.tokenizer.feature_extractor,
            # compute_metrics=self.compute_metrics,
            callbacks=callbacks,
            args = Seq2SeqTrainingArguments(
                per_device_train_batch_size = per_device_train_batch_size,
                gradient_accumulation_steps = gradient_accumulation_steps,
                warmup_steps = warmup_steps,
                
                learning_rate = learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = logging_steps,
                optim = optim,
                weight_decay = weight_decay,
                remove_unused_columns=remove_unused_columns,
                lr_scheduler_type = lr_scheduler_type,
                label_names = label_names,
                eval_steps = eval_steps,
                eval_strategy=eval_strategy,
                seed = seed,
                output_dir = output_dir,
                report_to = report_to , 
                **kwargs
            ),
        )
    
    def start_train(self):
        self._trainer.train()
        
    def save_model(
        self,output_dir="model_merged",
        save_method:Literal["merged_16bit","merged_4bit","lora"]="merged_16bit"):
        try:
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method = save_method)
        except:
            self.load_dataset(None)
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method = save_method)
    
    def predict(self,audio,**kwargs):
        try:model=self.model
        except:model=self.base_model
        
        FastModel.for_inference(model)
        model.eval()

        inputs = self.tokenizer(audio, return_tensors="pt",sampling_rate=16_000)
        input_features = inputs.input_features.to(device=model.device,dtype=model.dtype)
        generated_ids = model.generate(inputs=input_features)
        transcription = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription
       

if __name__ =="__main__":
    x=Whisper()
    x.load_model("model")

    dataset = load_dataset("FILM6912/STT-v2", split="train")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset=dataset.select(range(1000))
    dataset = dataset.train_test_split(test_size=0.006)
    train_dataset = dataset['train']
    test_dataset =dataset['test']
    
    x.load_dataset(train_dataset,test_dataset)
    
    x.triner(
        max_steps = 10,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        remove_unused_columns=False,
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        eval_steps = 5,
        eval_strategy="steps",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", 
        callbacks=None
        )
    
    x.start_train()
    
    x.save_model("model",save_method="merged_16bit")
    
