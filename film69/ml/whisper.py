from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, PeftModel, PeftConfig, LoraModel, LoraConfig, get_peft_model,prepare_model_for_kbit_training
# from peft import LoraConfig, PeftModel, PeftConfig, LoraModel, LoraConfig, get_peft_model,prepare_model_for_int8_training
from datasets import load_dataset, DatasetDict,Audio
from transformers import Seq2SeqTrainingArguments,pipeline
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os,gc,torch
import shutil
import evaluate

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control
    
class Whisper:
    def __init__(self) -> None:
        pass
    
    def prepare_dataset(self,batch):
        audio = batch["audio"]
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch
    
    def load_model(self,model_name,language = "Thai",task = "transcribe",load_in_4bit=False,max_new_tokens=128,device_map="auto",**kwargs):
        self.model_name=model_name
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
        self.base_model = WhisperForConditionalGeneration.from_pretrained(model_name, load_in_4bit=load_in_4bit, device_map=device_map,**kwargs)
        
        if not load_in_4bit:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.whisper = pipeline(
                "automatic-speech-recognition",
                model=self.base_model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=max_new_tokens,
                torch_dtype=torch_dtype,
                # device=device_map,
            )
        
    def load_dataset(self,train_dataset,test_dataset=None,num_proc=1):
        self.dataset=DatasetDict()
        self.dataset["train"]=train_dataset
        if test_dataset!=None:self.dataset["test"]=test_dataset
        try:self.dataset = self.dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"])
        except:pass
        self.dataset_after_map=self.dataset
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        self.dataset = self.dataset.map(self.prepare_dataset, remove_columns=self.dataset.column_names["train"], num_proc=num_proc)
        self.init_train()
        
    def init_train(self):
        self.metric = evaluate.load("wer")
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        # self.base_model = prepare_model_for_int8_training(self.base_model)
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        def make_inputs_require_grad(module, input, output):output.requires_grad_(True)
        self.base_model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        self.peft_model = get_peft_model(self.base_model, config)
        self.peft_model.print_trainable_parameters()
        

    def compute_metrics(self,pred):
        
        
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 *  self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


        
    def triner(self,output_dir="outputs",callbacks=None,**kwargs):
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            label_names=["labels"],
            **kwargs
        )
        self._trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.peft_model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"] if "test" in self.dataset.column_names.keys() else None,
            data_collator=self.data_collator,
            tokenizer=self.processor.feature_extractor,
            # callbacks=[SavePeftModelCallback],
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
    
    def start_train(self):
        self._trainer.train()
        
    def save_merged(self,output_dir="model_merged",low_vram=False,return_vram=False,save_method="bf16"):
        lora_adapter = "./lora-adapter"
        self.peft_model.save_pretrained(lora_adapter, save_adapter=True, save_config=True)
        if low_vram:
            del self._trainer
            del self.base_model
            del self.peft_model
            gc.collect()
            torch.cuda.empty_cache()
            
        if save_method == "bf64":
            save_method = torch.float64
        elif save_method== "bf32":
            save_method = torch.float32
        else:
            save_method = torch.float16  
        
        
        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=save_method,
            device_map="auto",
        )
        
        peft_model = PeftModel.from_pretrained(base_model, lora_adapter)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        # tokenizer.save_pretrained("./whisper-lora-merged")
        self.processor.save_pretrained(output_dir)
        if return_vram:
            del base_model
            del peft_model
            del merged_model
            del self.processor
            gc.collect()
            torch.cuda.empty_cache()
        shutil.rmtree(lora_adapter)
        
    def predict(self,audio,chunk_length_s=30,stride_length_s=5,batch_size=8,**kwargs):
        return self.whisper(audio,chunk_length_s=chunk_length_s,stride_length_s=stride_length_s,batch_size=batch_size,**kwargs)
    
            
            
def eval(trainer,task = "transcribe",language = "Thai"):
    import numpy as np
    from torch.utils.data import DataLoader
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
    import evaluate
    from tqdm import tqdm
        
    lora_adapter = "./eval-lora-adapter"
    trainer.peft_model.save_pretrained(lora_adapter, save_adapter=True, save_config=True)

    peft_model_id = lora_adapter
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, 
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    metric = evaluate.load("wer")
    eval_dataloader = DataLoader(trainer.dataset["test"], batch_size=8, collate_fn=trainer.data_collator)
    forced_decoder_ids = trainer.processor.get_decoder_prompt_ids(language=language, task=task)
    normalizer = BasicTextNormalizer()

    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, trainer.processor.tokenizer.pad_token_id)
                decoded_preds = trainer.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = trainer.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
            del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

    print(f"{wer=} and {normalized_wer=}")
    print(eval_metrics)
    shutil.rmtree(lora_adapter)

if __name__ =="__main__":
    dataset_name = "mozilla-foundation/common_voice_17_0"
    language_abbr = "th"
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", cache_dir="data")
    common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", cache_dir="data")
    print(common_voice)
    

    custom_train_size = 100
    custom_test_size = 20

    common_voice["train"] = common_voice["train"].select(range(custom_train_size))
    common_voice["test"] = common_voice["test"].select(range(custom_test_size))

    model_name_or_path = "whisper-large-v3-turbo"
    task = "transcribe"
    language = "Thai"
    trainer=Whisper()
    
    trainer.load_model(model_name_or_path,language,task)
    
    trainer.load_dataset(common_voice["train"],common_voice["test"],1)
    # trainer.load_dataset(common_voice["train"],None,1)
    
    trainer.triner(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
    
        # report_to=["tensorboard"],
        metric_for_best_model="wer",
        greater_is_better=False,
        # push_to_hub=True,
        max_steps=10,
        save_steps=10,
        eval_steps=10,
        logging_steps=1,

    )
    
    trainer.start_train()
    
    eval(trainer)
        
    
    trainer.save_merged(output_dir="model_merged",save_method="bf16")
    