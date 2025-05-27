import torch
from PIL import Image
# from transformers import ColPaliForRetrieval,AutoProcessor
from transformers import ColQwen2ForRetrieval,AutoProcessor
import numpy as np

class EmbeddingModel:
    def __init__(
        self,
        model_path = "vidore/colqwen2-v1.0-hf",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", 
        **kwargs
    ):
        self.model = ColQwen2ForRetrieval.from_pretrained(model_path,torch_dtype=torch_dtype,device_map=device_map, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path,use_fast=True)

        self.model=torch.compile(self.model)

    def __call__(self,text=None,image=None,return_np=True):
        if text is not None and image is not None:
            raise ValueError("Either text or image should be provided, not both.")
        elif text is not None:
            return self.text_embedding(text,return_np)
        elif image is not None:
            return self.image_embedding(image,return_np)
        else:
            raise ValueError("Either text or image should be provided.")

    def text_embedding(self,text:list[str],return_np=True):
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            embeddings = self.model(**inputs)
        return embeddings.embeddings.float().cpu().numpy().tolist() if return_np else embeddings.embeddings

    def image_embedding(self,image:list[Image.Image],return_np=True):
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            embeddings = self.model(**inputs)
        return embeddings.embeddings.float().cpu().numpy().tolist() if return_np else embeddings.embeddings

    def score_retrieval(self,query_embeddings,doc_embeddings):
        
        query_embeddings = torch.tensor(np.array(query_embeddings)).to(self.model.device,dtype=self.model.dtype)
        doc_embeddings = torch.tensor(np.array(doc_embeddings)).to(self.model.device,dtype=self.model.dtype)

        return self.processor.score_retrieval(query_embeddings,doc_embeddings).float().cpu().numpy().tolist()


if __name__ == '__main__':
    model = EmbeddingModel()
    text_embedding = model.text_embedding(["a photo of a cat"])
    print(text_embedding.shape)
    image = Image.open("cat.jpg")
    image_embedding = model.image_embedding([image])
    print(image_embedding.shape)