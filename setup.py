from setuptools import setup, find_packages

setup(
    name='film69',
    version='0.1.7',
    packages=find_packages(),
    install_requires=[
       "minimalmodbus",
       "transformers==4.42.3",
       "sentence-transformers==3.0.1",
       "llama-index-vector-stores-milvus==0.1.20",
       "llama-index==0.10.52",
       "llama-index-embeddings-huggingface==0.2.2",
       "pymilvus==2.4.4",
       "openai",
       "accelerate"
    ],
    author='Watcharaphon Pamayayang',
    author_email='filmmagic45@gmail.com',
    # description='คำอธิบายสั้นๆ ของแพ็กเกจ',
    # url='URL ของโปรเจกต์หากมี',
)
