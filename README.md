# FILM69
### install
```sh
pip install git+https://github.com/WATCHARAPHON6912/FILM69.git
```
### example
```python
from film69.ml.model import LLMModel
model=LLMModel("scb10x/typhoon-7b-instruct-02-19-2024")
for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")
print(model.generate("สวัสดี",max_new_tokens=200))
```
```python
from film69.ml.llm_rag import LlmRag_PromptEngineering
import ast
x=LlmRag_PromptEngineering("data.db",api_key="")
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
