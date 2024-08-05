from setuptools import setup, find_packages

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
        'peft',
        'bitsandbytes',
        'xformers<0.0.27',
        'trl<0.9.0',
        'unsloth @ git+https://github.com/unslothai/unsloth.git#egg=unsloth',
        # Add additional requirements if needed
    ],
    author='Watcharaphon Pamayayang',
    author_email='filmmagic45@gmail.com',
    # description='A brief description of the package',  # Update this with your package's description
    url='https://github.com/watcharaphon6912',  # Update this with your project's URL
)
