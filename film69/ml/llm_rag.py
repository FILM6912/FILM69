import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util
from pymilvus import MilvusClient,Collection
from openai import OpenAI
import json
from film69.ml.model import LLMModel
import random

class LlmRag_PromptEngineering:
    def __init__(self,database:str="data.db",embedding:str='kornwtp/SCT-KD-model-XLMR',model:str ="typhoon-v1.5-instruct",local:bool=False,api_key:str=None,is_chat:bool=False):
        self.model_vec = SentenceTransformer(embedding)
        self.database=database
        query_embedding = self.model_vec.encode("hi")
        self.vector_name="vector"
        self.text="text"
        self.collection_name="collection"
        self.local=local
        if not is_chat:
            self.client_db = MilvusClient(self.database)
            self.client_db.create_collection(
            collection_name= self.collection_name,
            dimension=int(len(query_embedding))
            )
        if self.local==True:self.model=LLMModel(model)
        else:
            if api_key!=None:
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


    def generate_unique_ids(self,existing_ids, num_ids, id_length=10):
        characters = '0123456789'
        new_ids = []
        while len(new_ids) < num_ids:
            new_id = ''.join(random.choices(characters, k=id_length))
            if new_id not in existing_ids and new_id not in new_ids:new_ids.append(int(new_id))
        return new_ids

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

    def model_generate(self,text,max_new_tokens=100,limit=1):
        data_text=""
        for i in self.query(text,limit):data_text+="\n"+i
        if self.local:
            return self.model.generate(self.create_prompt(text,data_text),stream=True,max_new_tokens=max_new_tokens)
        else:
            stream= self.client_api.chat.completions.create(
                model=self.model,
                messages=[{"role": "user","content": self.create_prompt(text,data_text)}],
                max_tokens=max_new_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None: yield chunk.choices[0].delta.content

    
    def get_data(self,filter="id >= 0"):
            try:
                keys=self.client_db.query(collection_name= self.collection_name,filter=filter,output_fields=["*"],limit=1)[0].keys()
                res = self.client_db.query(collection_name= self.collection_name,filter=filter,output_fields=[i for i in keys if i !=self.vector_name],)
            except:return [{"id":"No information"}]
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
            
    def create(self,data_update:dict[list]={"text":[]}):
        data=[]
        vec=self.model_vec.encode(data_update["text"])
        print(len(data_update["text"]))
        if "id" not in data_update.keys():
            dict_list = [ast.literal_eval(str(item)) for item in self.get_data()]
            ids=self.generate_unique_ids(pd.DataFrame(dict_list)["id"].values,int(len(data_update["text"])))
            data_update["id"]=ids
        for i in range(len(data_update["text"])):
            data.append({self.text: data_update["text"][i],self.vector_name:vec[i]})
            for j in data_update.keys():
                if j!="text":data[-1][j]=data_update[j][i]

        res = self.client_db.insert(collection_name= self.collection_name,data=data)
        return res
            
    def update(self,data_update:dict[list]={"id":[],"text":[]}):

        data=[]
        vec=self.model_vec.encode(data_update["text"])
        for i in range(len(data_update["text"])):
            data.append({self.text: data_update["text"][i],self.vector_name:vec[i]})
            for j in data_update.keys():
                if j!="text":data[-1][j]=data_update[j][i]
        res = self.client_db.upsert(collection_name= self.collection_name,data=data)
        return res
    
if __name__ == '__main__':
    x=LlmRag_PromptEngineering("data.db",api_key="sk-m7jPZrZO873FSLx5MHIjH6VEPEzCAtwRwYIXGNOH6KJiLC9i")
    # x.create({"text":["คุณคือ ai ที่สร่างโดย film","คุณคือผู้ช่วย"],
    #           "date":["55","66"]})
    
    print(x.query("สร่างโดย"))
    # print(x.get_data())

    # dict_list = [ast.literal_eval(str(item)) for item in x.get_data()]
    # print(pd.DataFrame(dict_list))
    print(x.create({
         "id":[1250834420,3771826426],
         "text":["สร่างโดย film69","คุณคือผู้ช่วย ai"],
         "create_date":["01","00"],
         "last_update":["01","00"],
         "diseases":["bdhjwed","djiqawhidshowq"]
         }))

    # print(x.update({
    #      "id":[1250834420,3771826426],
    #      "text":["สร่างโดย film69","คุณคือผู้ช่วย ai"],
    #      "date":["01","00"]
    #      }))
    dict_list = [ast.literal_eval(str(item)) for item in x.get_data()]
    print(pd.DataFrame(dict_list))


