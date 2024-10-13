from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model,prepare_model_for_int8_training
from datasets import load_dataset, DatasetDict,Audio
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os,gc,torch,evaluate
import shutil

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
    
    def load_model(self,model_name,language = "Thai",task = "transcribe",**kwargs):
        self.model_name=model_name
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
        self.base_model = WhisperForConditionalGeneration.from_pretrained(model_name, load_in_4bit=True, device_map="auto",**kwargs)
        
    def load_dataset(self,train_dataset,test_dataset=None,num_proc=1):
        self.dataset=DatasetDict()
        self.dataset["train"]=train_dataset
        if test_dataset!=None:self.dataset["test"]=test_dataset
        self.dataset = self.dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"])
        self.dataset_after_map=self.dataset
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        self.dataset = self.dataset.map(self.prepare_dataset, remove_columns=self.dataset.column_names["train"], num_proc=num_proc)
        self.init_train()
        
    def init_train(self):

        self.metric = evaluate.load("wer")
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        self.base_model = prepare_model_for_int8_training(self.base_model)
        def make_inputs_require_grad(module, input, output):output.requires_grad_(True)
        self.base_model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        self.peft_model = get_peft_model(self.base_model, config)
        self.peft_model.print_trainable_parameters()
        
    def compute_metrics(self,pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
        
    def triner(self,output_dir="whisper-large-v3-turbo-thai",callbacks=None,**kwargs):
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
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
            # callbacks=[SavePeftModelCallback],
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
        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if save_method == "bf16" else None,
            load_in_4bit=True if save_method == "bf4" else False,
            load_in_8bit=True if save_method == "bf8" else False,
        ).to("cuda")
        
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
    
    triner=Whisper()
    triner.load_model(model_name_or_path,language,task)
    
    triner.load_dataset(common_voice["train"],common_voice["test"],1)
    # triner.load_dataset(common_voice["train"],None,1)
    
    triner.triner(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=50,
        evaluation_strategy="no",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=10,
        max_steps=100,
    )
    
    triner.start_train()
    triner.save_merged(output_dir="model_merged",low_vram=False,return_vram=False)
    