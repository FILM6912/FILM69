# FILM69
### install
```sh
pip install git+https://github.com/WATCHARAPHON6912/FILM69.git
```
### example
```python
from film69.ml.localmodel import LocalModel
model=LocalModel("scb10x/typhoon-7b-instruct-02-19-2024")
for text in model.generate("สวัสดี",stream=True,max_new_tokens=200):
    print(text,end="")
print(model.generate("สวัสดี",max_new_tokens=200))
```

