# FILM69
### install
```sh
pip install git+https://github.com/watcharaphon6912/film69.git#egg=film69[all]
```
```sh
pip install git+https://github.com/watcharaphon6912/film69.git#egg=film69[rag]
```

### example
#### FastLLM
```python
from film69.ml.fast_model import FastLLM
model=FastLLM()
model.load_model(
    "FILM6912/XiaoXi-TH-8B",
    device_map="cuda",
    load_in_4bit=True,
    # load_in_8bit=True,
    # low_cpu_mem_usage = True
    )
for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")
print(model.generate("สวัสดี",max_new_tokens=200))
```

#### LLM
```python
from film69.ml.model import LLMModel
model=LLMModel(
    "FILM6912/XiaoXi-TH-8B",
    device_map="cuda",
    load_in_4bit=True,
    # load_in_8bit=True,
    # low_cpu_mem_usage = True
)
for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")
print(model.generate("สวัสดี",max_new_tokens=200))
```

#### LLM API
```python
from film69.ml.model import LLMModel
from openai import OpenAI
api=OpenAI(api_key="your_api_key",base_url="https://api.opentyphoon.ai/v1",)
model=LLMModel(api=api,model_name="typhoon-v1.5x-70b-instruct",local=False)
for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")
print(model.generate("สวัสดี",max_new_tokens=200))
```

#### VectorDB
```python
from film69.ml.vectordb import VectorDB
db = VectorDB()
db.add_or_update(
    documents=["สวัสดี","ไปไหน"],
    ids=["doc1","doc2"],
    metadatas=[{"da":"ทักทาย","ff":"hi"},{"da":"คำถาม","ff":"question"}],
)
db.query(query_texts=["Where are you going?"])
db.get()
db.delete(["doc1","doc2"])
```

#### RAG+PromptEngineering
```python
from film69.ml.llm_rag import LlmRag_PromptEngineering
import ast
x=LlmRag_PromptEngineering("data.db",api_key="")
x.prompt_engineering="""
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
x.create({"text":["คุณคือผู้ช่วย"],
               "date":["55"]})
    
print(x.query("คุณคือ"))
print(x.get_data())

dict_list = [ast.literal_eval(str(item)) for item in x.get_data()]
print(pd.DataFrame(dict_list))
print(x.create({
         "id":[1250834420,3771826426],
         "text":["คุณคือผู้ช่วย ai"],
         "last_update":["01"],
         }))

print(x.update({
         "id":[1250834420,3771826426],
          "text":["คุณคือผู้ช่วย ai"],
          "last_update":[,"01"],
          }))
dict_list = [ast.literal_eval(str(item)) for item in x.get_data()]
print(pd.DataFrame(dict_list))

```
