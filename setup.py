from setuptools import setup, find_packages

setup(
    name='film69',
    version='0.2.3',
    packages=find_packages(),
    install_requires=[
       "minimalmodbus",
       "transformers",
       "sentence-transformers",
       "llama-index-vector-stores-milvus",
       "llama-index==0.10.52",
       "llama-index-embeddings-huggingface",
       "pymilvus",
       "openai",
       "accelerate",
       "git+https://github.com/unslothai/unsloth.git",
       "datasets",
       "xformers",
       "trl",
       "peft",
       "bitsandbytes"
    ],
    author='Watcharaphon Pamayayang',
    author_email='filmmagic45@gmail.com',
    # description='คำอธิบายสั้นๆ ของแพ็กเกจ',
    # url='URL ของโปรเจกต์หากมี',
)
