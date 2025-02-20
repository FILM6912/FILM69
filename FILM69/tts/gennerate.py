import numpy as np

from FILM69.tts.f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from FILM69.tts.f5_tts.model import DiT, UNetT
import torch

class TTS:
    def __init__(self,
            ckpt_path="model_300.safetensors",
            vocab_file="vocab.txt",
            dim=1024,
            depth=22,
            heads=16, 
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            device="auto"
            ):
        self.model=self.load_model( ckpt_path=ckpt_path,vocab_file=vocab_file,dim=dim, depth=depth, heads=heads, ff_mult=ff_mult, text_dim=text_dim, conv_layers=conv_layers,device=device)
        self.vocoder = load_vocoder()
        
    def load_model(self,ckpt_path="model_300.safetensors",vocab_file="vocab.txt",dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,device="auto"):
        model_cfg = dict(dim=dim, depth=depth, heads=heads, ff_mult=ff_mult, text_dim=text_dim, conv_layers=conv_layers)
        
        if device=="auto":
            device="cuda" if torch.cuda.is_available() else "cpu"
        
        return load_model(DiT, model_cfg, ckpt_path,vocab_file=vocab_file,device=device)
    
    def generate(self,ref_audio,ref_text:str,gen_text:str):
        vocoder_name = "vocos"

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
                    speed=1,
                    fix_duration=fix_duration,
                )
        
        return audio_segment,sr,spectragram