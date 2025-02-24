from setuptools import setup, find_packages

# Common packages used across multiple extras
common_packages = [
    "setuptools",
    "transformers>=4.44.2",
    "numpy",
    "pandas",
    "openpyxl",
    "openai",
    "sentence-transformers",
    "ipywidgets",
    "plotly",
    "matplotlib"
]

LLM=[
    "datasets",
    "sentencepiece",
    "tqdm",
    "psutil",
    "accelerate",
    "trl",
    "peft",
    "huggingface-hub",
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
]

ubuntu=[
    "flash-attn==2.7.3",
    "xformers",
    "ninja",
    # "bitsandbytes",
]

RAG=[
    "pymilvus",
    "chromadb",
]

UI=[
    "streamlit",
    "stqdm",
    "ipywidgets",
    "gradio",
]

SPEECH=[
    "datasets>=2.6.1",
    "librosa",
    "evaluate==0.4.3",
    "jiwer",
    "accelerate",
    "cached_path",
    "ema_pytorch>=0.5.2",
    "hydra-core>=1.3.0",
    "jieba",
    "pydub",
    "pypinyin",
    "safetensors",
    "soundfile",
    "tomli",
    "torchaudio",
    "torchdiffeq",
    "transformers_stream_generator",
    "vocos",
    "wandb",
    "x_transformers>=1.31.14",
    "click",
    
]
IOT=["minimalmodbus"]
langchain=[
    "langchain",
    "langchain-ollama",
    "langchain-openai",
    "langgraph",
    "langchain-core"
]

setup(
    name="film69",
    version="0.4.8",
    author="Watcharaphon Pamayayang",
    author_email="filmmagic45@gmail.com",
    url="https://github.com/watcharaphon6912",
    packages=find_packages(),
    python_requires=">=3.10",
     entry_points={
        "console_scripts": [
            "tts_used_cli=FILM69.tts.f5_tts.infer.infer_cli:main",
            "tts_used_gradio=FILM69.tts.f5_tts.infer.infer_gradio:main",
            "tts_train_cli=FILM69.tts.f5_tts.train.finetune_cli:main",
            "tts_train_ui=FILM69.tts.f5_tts.train.finetune_gradio:main",
        ],
    },
    extras_require={
        "llm": common_packages + LLM,
        "rag": common_packages + RAG,
        "speech": common_packages + SPEECH,
        "ui": UI,
        "iot":IOT,
        "all": common_packages + ubuntu + LLM + RAG + UI + SPEECH+IOT+langchain,
        "all_win": common_packages + LLM + RAG + UI + SPEECH+IOT+langchain,
        "all_llama-cpp": common_packages + LLM + RAG + UI + SPEECH +langchain+ ["llama-cpp-python==0.3.1"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
 