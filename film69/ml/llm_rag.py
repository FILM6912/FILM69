import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util
from pymilvus import MilvusClient,Collection
from openai import OpenAI
import json
from film69.ml.localmodel import LocalModel

class LlmRag_PromptEngineering:
    def __init__(self,database:str="data.db",embedding:str='kornwtp/SCT-KD-model-XLMR',model:str ="typhoon-v1.5-instruct",local:bool=False,api_key:str=None):
        self.model_vec = SentenceTransformer(embedding)
        self.database=database
        self.client_db = MilvusClient(self.database)
        self.vector_name="vector"
        self.text="text"
        self.collection_name="collection"
        self.local=local
        if self.local==True:self.model=LocalModel(model)
        else:
            self.client_api = OpenAI(
                api_key=api_key,
                base_url="https://api.opentyphoon.ai/v1",)
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

    def model_generate(self,text,max_new_tokens=100):
        if self.local:
            for text in self.model.generate(self.create_prompt(text,self.query(text,1)[0]),stream=True,max_new_tokens=max_new_tokens):
                yield text
        else:
            stream=self.client_api.chat.completions.create(
                model=self.model,
                messages=[{"role": "user","content": self.create_prompt(text,self.query(text,1)[0])}],
                max_tokens=512*5,
                temperature=0.6,
                top_p=1,
                stream=True,
            )
            text=""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    text=chunk.choices[0].delta.content
                    yield text
            yield text

    def get_data(self,filter="id >= 0"):
            keys=self.client_db.query(collection_name= self.collection_name,filter=filter,output_fields=["*"],limit=1)[0].keys()
            res = self.client_db.query(collection_name= self.collection_name,filter=filter,output_fields=[i for i in keys if i !=self.vector_name],)
            return res
    
    def delete(self,id:list[int]):
         self.client_db.delete(
        collection_name=self.collection_name,
        ids=id
)

    def query(self,text,limit=1):
            query_embedding = self.model_vec.encode(text)
            res = self.client_db.search(collection_name= self.collection_name,data=[query_embedding],output_fields=["id",self.text],limit=limit)
            return [i["entity"][self.text] for i in res[0]]
            
    def update(self,data_update:dict[list]={"text":[]}):
        data=[]
        vec=self.model_vec.encode(data_update["text"])
        for i in range(len(data_update["text"])):
            data.append({self.text: data_update["text"][i],self.vector_name:vec[i]})
            for j in data_update.keys():
                if j!="text":data[-1][j]=data_update[j][i]

        self.client_db.create_collection(
        collection_name= self.collection_name,
        auto_id=True,
        dimension=int(len(data[0][self.vector_name]))
        )
        res = self.client_db.insert(collection_name= self.collection_name,data=data)
        return res
    
if __name__ == '__main__':
    x=LlmRag_PromptEngineering("data/data.db")
    # x.update({"text":["คุณคือ ai ที่สร่างโดย film","คุณคือผู้ช่วย"],
    #           "date":["55","66"]})
    print(x.query("คุณคือ"))
    # dict_list = [ast.literal_eval(str(item)) for item in x.get()]
    # print(pd.DataFrame(dict_list))

