from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]

setup(
    name='film69',
    version='0.2.3',
    packages=find_packages(),
    install_requires=[
    'minimalmodbus',
    'transformers',
    'sentence-transformers',
    'llama-index-vector-stores-milvus',
    'llama-index==0.10.52',
    'llama-index-embeddings-huggingface',
    'pymilvus',
    'openai',
    'accelerate',
    'datasets',
    'xformers',
    'trl',
    'peft',
    'bitsandbytes'
],
    dependency_links=[
        'git+https://github.com/unslothai/unsloth.git#egg=unsloth',
    ],
    author='Watcharaphon Pamayayang',
    author_email='filmmagic45@gmail.com',
    # description='A brief description of the package',
    # url='URL of the project if available',
)
