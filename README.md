# FILM69
### install
```sh
pip install git+https://github.com/watcharaphon6912/film69.git#egg=film69[all]
```
```sh
pip install git+https://github.com/watcharaphon6912/film69.git#egg=film69[rag]
```

### example
#### LLM
```python
from film69.ml.model import LLMModel
model=LLMModel(
    "scb10x/typhoon-7b-instruct-02-19-2024",
    device_map="cuda",
    load_in_4bit=True,
    # load_in_8bit=True,
    # low_cpu_mem_usage = True
)
for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")
print(model.generate("สวัสดี",max_new_tokens=200))
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
    x.create({"text":["คุณคือ ai ที่สร่างโดย film","คุณคือผู้ช่วย"],
               "date":["55","66"]})
    
    print(x.query("สร่างโดย"))
    print(x.get_data())

    dict_list = [ast.literal_eval(str(item)) for item in x.get_data()]
    print(pd.DataFrame(dict_list))
    print(x.create({
         "id":[1250834420,3771826426],
         "text":["สร่างโดย film69","คุณคือผู้ช่วย ai"],
         "last_update":["01","00"],
         }))

    print(x.update({
         "id":[1250834420,3771826426],
          "text":["สร่างโดย film69","คุณคือผู้ช่วย ai"],
          "last_update":["02","01"],
          }))
    dict_list = [ast.literal_eval(str(item)) for item in x.get_data()]
    print(pd.DataFrame(dict_list))

```
