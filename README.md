# FILM69

<details>
  <summary style="font-size: 25px;">Install</summary>

```sh
pip install "git+https://github.com/watcharaphon6912/film69.git@v0.4.7#egg=film69[all]"
```
```sh
pip install "git+https://github.com/watcharaphon6912/film69.git#egg=film69[llm]"
```
```sh
pip install "git+https://github.com/watcharaphon6912/film69.git#egg=film69[rag]"
```
```sh
pip install "git+https://github.com/watcharaphon6912/film69.git#egg=film69[speech]"
```
```sh
pip install "git+https://github.com/watcharaphon6912/film69.git#egg=film69[ui]"
```
```sh
pip install "git+https://github.com/watcharaphon6912/film69.git#egg=film69[iot]"
```
```sh
pip install "git+https://github.com/watcharaphon6912/film69.git#egg=film69[all_llama-cpp]"
```
</details>

```sh
pip install "git+https://github.com/watcharaphon6912/film69.git#egg=film69" --force-reinstall
```


# example
#### FastAutoModel
```python
from FILM69.llm import FastAutoModel
from PIL import Image

image=Image.open("image.jpg")

model=FastAutoModel(
    "FILM6912/Llama-3.2-11B-Vision-Instruct",
    device_map="cuda",
    load_in_4bit=True,
    )

for i in model.generate("คุณเห็นอะไรในรูป",image,stream=True):
    print(i,end="")

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("คุณเห็นอะไรในรูป",image,max_new_tokens=200))
print(model.generate("สวัสดี",max_new_tokens=200))


#####################################################################################

model=FastAutoModel(
    "FILM6912/Llama-3.2-1B-Instruct",
    device_map="cuda",
    load_in_4bit=True,
    )

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("สวัสดี",max_new_tokens=200))

```



#### FastVLLM
```python
from FILM69.llm import FastVLLM
from PIL import Image

image=Image.open("image.jpg")

model=FastVLLM()
model.load_model(
    "FILM6912/Llama-3.2-11B-Vision-Instruct",
    device_map="cuda",
    load_in_4bit=True,
    )

for i in model.generate("คุณเห็นอะไรในรูป",image,history_save=False,stream=True):
    print(i,end="")

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("สวัสดี",max_new_tokens=200))
```


#### FastLLM
```python
from FILM69.llm import FastLLM
model=FastLLM()
model.load_model(
    "FILM6912/Llama-3.2-1B-Instruct",
    device_map="cuda",
    load_in_4bit=True,
    )

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("สวัสดี",max_new_tokens=200))
```


<details>
  <summary style="font-size: 15px;">LLM</summary>

```python
from FILM69.llm.model import LLMModel
model=LLMModel(
    "FILM6912/Llama-3.2-1B-Instruct",
    device_map="cuda",
    load_in_4bit=True,
    # load_in_8bit=True,
    # low_cpu_mem_usage = True
    )

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("สวัสดี",max_new_tokens=200))
```
</details>


<details>
  <summary style="font-size: 15px;">LLM API</summary>

```python
from FILM69.llm.model import LLMModel
from openai import OpenAI

api=OpenAI(api_key="your_api_key",base_url="https://api.opentyphoon.ai/v1",)
model=LLMModel(api=api,model_name="typhoon-v1.5x-70b-instruct",local=False)

for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")

print(model.generate("สวัสดี",max_new_tokens=200))
```
</details>


<details>
  <summary style="font-size: 15px;">VectorDB</summary>

```python
from FILM69.llm.vectordb import VectorDB

db = VectorDB()
db.add_or_update(
    documents=["สวัสดี","ไปไหน"],
    ids=["doc1","doc2"],
    metadatas=[{"type":"ทักทาย"},{"type":"คำถาม"}],
    )

db.query(query_texts=["Where are you going?"])
db.get()
db.delete(["doc1","doc2"])
```
</details>


<details>
  <summary style="font-size: 15px;">RAG Chromadb</summary>

```python
from FILM69.llm.llm_rag_chromadb import LlmRagChromadb
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

```
</details>