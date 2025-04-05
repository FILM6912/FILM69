import os, requests
from cached_path import cached_path
from FILM69.tts.f5_tts.train.finetune_gradio import expand_model_embeddings
import argparse
import os
import shutil
from FILM69.tts.f5_tts.model import CFM, UNetT, DiT, Trainer
from FILM69.tts.f5_tts.model.utils import get_tokenizer
from FILM69.tts.f5_tts.model.dataset import load_dataset
from importlib.resources import files
from datasets import load_dataset as _load_dataset,Audio as _Audio,concatenate_datasets
import numpy as np
from tqdm.autonotebook import trange

from FILM69.tts.f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    # speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
import torch


class TTS:
    def __init__(self):
        self.datasets=None
        self.model=None
    
    def load_model(self, 
            ckpt_path="model_300.safetensors",
            vocab_file="vocab.txt",
            dim=1024,
            depth=22,
            heads=16, 
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            device="auto"):
        self.model=self._load_model( ckpt_path=ckpt_path,vocab_file=vocab_file,dim=dim, depth=depth, heads=heads, ff_mult=ff_mult, text_dim=text_dim, conv_layers=conv_layers,device=device)
        self.vocoder = load_vocoder()
    
    def _load_model(self,ckpt_path="model_300.safetensors",vocab_file="vocab.txt",dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,device="auto"):
        model_cfg = dict(dim=dim, depth=depth, heads=heads, ff_mult=ff_mult, text_dim=text_dim, conv_layers=conv_layers)
        if device=="auto":
            device="cuda" if torch.cuda.is_available() else "cpu"
        
        return load_model(DiT, model_cfg, ckpt_path,vocab_file=vocab_file,device=device)
        
    def predict(self,ref_audio,ref_text:str,gen_text:str,vocoder_name = "vocos",speed=1):
        audio_segment, sr, spectragram = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            self.model,
            self.vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
        )
        
        return audio_segment,sr,spectragram
        
    def load_datasets(self,datasets,text_column="text",audio_column="audio",duration_column=None):
        self.datasets=datasets
        self.text_column=text_column
        self.audio_column=audio_column
        self.duration_column=duration_column
        
    def vocab_check(self,datasets,file_vocab):
        with open(file_vocab, "r", encoding="utf-8-sig") as f:
            data = f.read()
            vocab = data.split("\n")
            vocab = set(vocab)

        miss_symbols = []
        miss_symbols_keep = {}
        for i in trange(len(datasets),desc="Checking vocab"):
            text=datasets[i]["text"]
            for t in text:
                if t not in vocab and t not in miss_symbols_keep:
                    miss_symbols.append(t)
                    miss_symbols_keep[t] = t

        if miss_symbols == []:
            vocab_miss = ""
            info = "You can train using your language !"
        else:
            vocab_miss = ",".join(miss_symbols)
            info = f"The following symbols are missing in your language {len(miss_symbols)}\n\n"

        return info, vocab_miss
    
    def vocab_extend(self,output, symbols, model_type = "F5-TTS"):
        if symbols == "":
            return "Symbols empty!"
        
        file_vocab_project = f"{output}/new_vocab.txt"

        file_vocab = f"{output}/old_vocab.txt"
        if not os.path.isfile(file_vocab):
            return f"the file {file_vocab} not found !"

        symbols = symbols.split(",")
        if symbols == []:
            return "Symbols to extend not found."

        with open(file_vocab, "r", encoding="utf-8-sig") as f:
            data = f.read()
            vocab = data.split("\n")
        vocab_check = set(vocab)

        miss_symbols = []
        for item in symbols:
            item = item.replace(" ", "")
            if item in vocab_check:
                continue
            miss_symbols.append(item)

        if miss_symbols == []:
            return "Symbols are okay no need to extend."

        size_vocab = len(vocab)
        vocab.pop()
        for item in miss_symbols:
            vocab.append(item)

        vocab.append("")

        with open(file_vocab_project, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab))

        if model_type == "F5-TTS":
            ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
        else:
            ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))

        vocab_size_new = len(miss_symbols)

        new_ckpt_path = output+"/checkpoints"
        os.makedirs(new_ckpt_path, exist_ok=True)

        new_ckpt_file = os.path.join(new_ckpt_path, "pretrained_model_1200000.pt")

        size = expand_model_embeddings(ckpt_path, new_ckpt_file, num_new_tokens=vocab_size_new)

        vocab_new = "\n".join(miss_symbols)
        return f"vocab old size : {size_vocab}\nvocab new size : {size}\nvocab add : {vocab_size_new}"
                
    def trainer(self,
        output="train_tts",
        model_type = "F5-TTS",
        exp_name='F5TTS_Base',
        dataset_name='tha',
        learning_rate=1e-05,
        batch_size_per_gpu=50000,
        batch_size_type='frame',
        max_samples=1,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        
        # epochs=1000,
        # save_per_updates=3*500,
        # last_per_updates=3*500,
        epochs=10,
        save_step=None,
        save_epochs=1,
        
        num_warmup_updates=0,
        keep_last_n_checkpoints=5,
        finetune=True,
        pretrain=None,
        tokenizer='char',
        tokenizer_path=None,
        log_samples=True,
        logger='tensorboard',
        bnb_optimizer=True,
        
        target_sample_rate = 24000,
        n_mel_channels = 100,
        hop_length = 256,
        win_length = 1024,
        n_fft = 1024,
        mel_spec_type = "vocos",  # 'vocos' or 'bigvgan'
        
        check_vocab=True,
        ):
        if save_step != None:
            save_per_updates=save_step
            last_per_updates=save_step
        else:
            save_per_updates=len(self.datasets)*save_epochs
            last_per_updates=len(self.datasets)*save_epochs
        
        os.makedirs(output+"/checkpoints", exist_ok=True)
        if not os.path.exists(f"{output}/old_vocab.txt"):
            os.makedirs(os.path.dirname(f"{output}/old_vocab.txt"), exist_ok=True)
            with open(f"{output}/old_vocab.txt", "wb") as f:
                f.write(requests.get("https://raw.githubusercontent.com/WATCHARAPHON6912/FILM69/refs/heads/main/data/Emilia_ZH_EN_pinyin/vocab.txt").content)
        
        if check_vocab:
            info,new_vocab=self.vocab_check(self.datasets,f"{output}/old_vocab.txt")
            info_extend=self.vocab_extend(output,new_vocab,model_type)
            print(info_extend)
        else:print("Not check vocab")
        
        os.environ["path"]=output
        
        # -------------------------- Dataset Settings --------------------------- #
        
        checkpoint_path = output+"/checkpoints"
        if exp_name == "F5TTS_Base":
                wandb_resume_id = None
                model_cls = DiT
                model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
                if finetune:
                    if pretrain is None:
                        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
                    else:
                        ckpt_path = pretrain
        elif exp_name == "E2TTS_Base":
                wandb_resume_id = None
                model_cls = UNetT
                model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
                if finetune:
                    if pretrain is None:
                        ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
                    else:
                        ckpt_path = pretrain

        if finetune:
                if not os.path.isdir(checkpoint_path):
                    os.makedirs(checkpoint_path, exist_ok=True)

                file_checkpoint = os.path.basename(ckpt_path)
                if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
                    file_checkpoint = "pretrained_" + file_checkpoint
                file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
                if not os.path.isfile(file_checkpoint):
                    shutil.copy2(ckpt_path, file_checkpoint)
                    print("copy checkpoint for finetune")

            # Use the tokenizer and tokenizer_path provided in the command line arguments
        tokenizer = tokenizer
        if tokenizer == "custom":
                if not tokenizer_path:
                    raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
                tokenizer_path = tokenizer_path
        else:
                tokenizer_path = dataset_name

        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
        print("\nvocab : ", vocab_size)
        print("\nvocoder : ", mel_spec_type)
        
        mel_spec_kwargs = dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

        self.model = CFM(
                transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
                mel_spec_kwargs=mel_spec_kwargs,
                vocab_char_map=vocab_char_map,
            )

        self._trainer = Trainer(
                self.model,
                epochs,
                learning_rate,
                num_warmup_updates=num_warmup_updates,
                save_per_updates=save_per_updates,
                keep_last_n_checkpoints=keep_last_n_checkpoints,
                checkpoint_path=checkpoint_path,
                batch_size=batch_size_per_gpu,
                batch_size_type=batch_size_type,
                max_samples=max_samples,
                grad_accumulation_steps=grad_accumulation_steps,
                max_grad_norm=max_grad_norm,
                logger=logger,
                wandb_project=dataset_name,
                wandb_run_name=exp_name,
                wandb_resume_id=wandb_resume_id,
                log_samples=log_samples,
                last_per_updates=last_per_updates,
                bnb_optimizer=bnb_optimizer,
            )
        self.train_dataset = load_dataset(
            self.datasets,
            tokenizer,
            mel_spec_kwargs=mel_spec_kwargs,
            dataset_type="HFDataset",
            text_column=self.text_column,
            audio_column=self.audio_column,
            duration_column=self.duration_column,
            )
        
    def start_train(self, resumable_with_seed=666):
        self._trainer.train(
            self.train_dataset,
            resumable_with_seed=resumable_with_seed,
        )
        

if __name__ == "__main__":
    x=TTS()
    
    data=load_dataset("FILM6912/STT-v2")
    data=data.rename_columns({"sentence":"text"})
    data=concatenate_datasets([data["train"],data["test"]])
    data=data.cast_column("audio",_Audio(sampling_rate=24000))
    
    x.load_datasets(data)
    
    x.trainer(
        exp_name='F5TTS_Base',
        learning_rate=1e-05,
        batch_size_per_gpu=10000, #24GB Vram
        batch_size_type='frame',
        max_samples=64,
        grad_accumulation_steps=1,
        max_grad_norm=1,
        epochs=20,
        num_warmup_updates= 405764,
        save_step=811528,
        keep_last_n_checkpoints=5,
        last_per_updates=10000,
        finetune=True,
        check_vocab=False
    )
    
    x.start_train()