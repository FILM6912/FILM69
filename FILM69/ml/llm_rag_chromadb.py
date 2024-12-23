
from openai import OpenAI
from .vectordb import VectorDB

class LlmRagChromadb(VectorDB):
    def __init__(self,
            path="database",
            collection_name="data",
            embedding_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            model:str ="typhoon-v1.5-instruct",
            local:bool=False,
            api_key:str=None,
            **kwargs
            ):
        super().__init__(path, collection_name, embedding_name)

        self.local=local
        if self.local==True:
            from .fast_model import FastLLM
            self.model=FastLLM()
            self.model.load_model(model,**kwargs)
        else:
            if type(api_key)==str:
                self.client_api = OpenAI(
                api_key=api_key,
                base_url="https://api.opentyphoon.ai/v1",)
            else:
                self.client_api=api_key
            self.model=model
        self.prompt_engineering="""
คุณกำลังเป็นผู้ช่วย AI ที่มีความเชี่ยวชาญในการตอบคำถามเกี่ยวกับข้อมูล โดยข้อมูลที่คุณจะใช้ในการตอบคำถามประกอบไปด้วย:

เนื้อหาต่างๆ

ให้คุณตอบคำถามตามข้อมูลที่ได้รับด้วยความแม่นยำสูงสุด โดยมีรายละเอียดดังนี้:

### คำถาม:
{question}

### ข้อมูลที่เกี่ยวข้อง:
{data}

### คำตอบ:
กรุณาให้คำตอบที่แม่นยำและครอบคลุมทุกแง่มุมของคำถามที่ผู้ใช้ถาม โดยใช้ข้อมูลที่ให้ไว้ด้านบน โดยตอบแค่ที่ถามเท่านั้น
"""

    def create_prompt(self,question,data):
        return self.prompt_engineering.replace("{question}",question).replace("{data}",data)
    
    def model_chat(self,text,max_new_tokens=100):
        if self.local:
            return self.model.generate(text,stream=True,max_new_tokens=max_new_tokens)
        else:
            stream= self.client_api.chat.completions.create(
                model=self.model,
                messages=[{"role": "user","content": text}],
                max_tokens=max_new_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None: yield chunk.choices[0].delta.content

    def model_generate(self,text,max_new_tokens=100,limit=1,stream=False,text_out="document",where=None):
        data_text=""
        for i in list(self.query(query_texts=text,on_dict=True,n_results=limit,where=where)[text_out].values()):data_text+="\n"+i
        if self.local:
            return self.model.generate(self.create_prompt(text,data_text),stream=stream,max_new_tokens=max_new_tokens,history_save=False)
        else:
            out= self.client_api.chat.completions.create(
                model=self.model,
                messages=[{"role": "user","content": self.create_prompt(text,data_text)}],
                max_tokens=max_new_tokens,
                stream=stream,
                )
            def inner():
                for chunk in out:
                    if chunk.choices[0].delta.content is not None: yield chunk.choices[0].delta.content
   
            if stream:return inner()
            else:return out.choices[0].message.content

if __name__ == "__main__":
    x=LlmRagChromadb(
        api_key="",
        model="typhoon-v1.5-instruct",
        local=False
    )

    x.add_or_update(
        ids=['1','2'],
        # ids=None,
        documents=["คุณคือ","สวัสดี"],
        metadatas=[{"text_out":"ฉันเป็น AI ที่ถูกสร้างขึ้นมาเพื่อช่วยตอบคำถาม","x":0},{"text_out":"สวัสดีครับ","x":1}]
    )
    
    print(x.query(query_texts="คุณคือ",n_results=10))
    
    print(x.get())
    
    for i in x.model_generate("คุณคือ",limit=1,text_out="text_out",where={"x":0},stream=True):
        print(i,end="")