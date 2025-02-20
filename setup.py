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

linux=[
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
    "gradio"
]

SPEECH=[
    "datasets>=2.6.1",
    "librosa",
    "evaluate==0.4.3",
    "jiwer",
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
            "f5-tts_infer-cli=FILM69.tts.f5_tts.infer.infer_cli:main",
            "f5-tts_infer-gradio=FILM69.tts.f5_tts.infer.infer_gradio:main",
            "f5-tts_finetune-cli=FILM69.tts.f5_tts.train.finetune_cli:main",
            "f5-tts_finetune-gradio=FILM69.tts.f5_tts.train.finetune_gradio:main",
        ],
    },
    extras_require={
        "LLM": common_packages + LLM,
        "rag": common_packages + RAG,
        "speech": common_packages + SPEECH,
        "ui": UI,
        "iot":IOT,
        "all": common_packages + LLM + RAG + UI + SPEECH+IOT+linux+langchain,
        "all_win": common_packages + LLM + RAG + UI + SPEECH+IOT+langchain,
        "all_llama-cpp": common_packages + LLM + RAG + UI + SPEECH +langchain+ ["llama-cpp-python==0.3.1"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
 