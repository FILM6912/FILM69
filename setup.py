from setuptools import setup, find_packages

setup(
    name='film69',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[],
    extras_require={
        'all':[
            'setuptools',
            'setuptools-scm',
            'packaging',
            'tyro',
            'transformers>=4.44.2',
            'datasets',
            'sentencepiece',
            'tqdm',
            'psutil',
            'wheel',
            'numpy',
            'accelerate',
            'trl',
            'peft',
            'protobuf',
            'huggingface-hub',
            'hf-transfer',
            'bitsandbytes',
            'xformers',
            'ninja',
            'minimalmodbus',
            'sentence-transformers',
            'llama-index-vector-stores-milvus',
            'llama-index',
            'llama-index-embeddings-huggingface',
            'pymilvus',
            'openai',
    ],
        'rag': [
            'pymilvus',
            'openai',
            'transformers>=4.44.2',
            'sentence-transformers',
            'numpy',
            'pandas'
        ],
    },
    author='Watcharaphon Pamayayang',
    author_email='filmmagic45@gmail.com',
    # description='A brief description of the package',  # Update this with your package's description
    url='https://github.com/watcharaphon6912',  # Update this with your project's URL
)