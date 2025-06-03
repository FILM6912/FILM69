from datasets import load_dataset, Audio
from transformers import MimiModel, AutoFeatureExtractor
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Encodec:
    def __init__(self, model_name: str="FILM6912/encodec_24khz", device: str="cpu",dtype=torch.float16,**kwargs):
        self.model_name = model_name
        self.feature_extractor_name = model_name
        self.device = device
        self.dtype=dtype
        self.model = MimiModel.from_pretrained(self.model_name,torch_dtype=dtype,**kwargs).to(device).float()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.sampling_rate=self.feature_extractor.sampling_rate

    def encode(self, audio_array):
        inputs = self.feature_extractor(raw_audio=audio_array, sampling_rate=self.sampling_rate, return_tensors="pt").to(self.device)
        inputs["padding_mask"]=inputs["padding_mask"].to(dtype=self.dtype)
        return self.model.encode(inputs["input_values"]).audio_codes

    def decode(self, encoded_values):
        return self.model.decode(encoded_values)[0].squeeze().detach().cpu().numpy()[:-1024]

