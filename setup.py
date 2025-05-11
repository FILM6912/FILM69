import os
import shutil
import importlib.util
from setuptools.command.install import install
from setuptools import setup, find_packages


class InstallCommand(install):
    def run(self):
        install.run(self)

        try:
            spec = importlib.util.find_spec("unsloth_zoo.saving_utils")
            if spec is None:
                print("unsloth_zoo.saving_utils not found. Skipping file copy.")
                return
            source = spec.origin

            spec_mod = importlib.util.find_spec("FILM69.llm.unsloth_zoo.saving_utils")
            if spec_mod is None:
                print("FILM69.llm.unsloth_zoo.saving_utils not found. Skipping file copy.")
                return
            destination = spec_mod.origin

            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(source, destination)
            print(f"Copied {source} to {destination}")
        except Exception as e:
            print("Error during post-install copy:", e)


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
    "matplotlib",
    "cached-path",
    "gradio",
    "streamlit",
    "flet"
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
    "langchain-core",
    "unsloth",
    "unsloth-zoo @ git+https://github.com/rupaut98/unsloth-zoo.git@fix-gemma-vision"
]

ubuntu=[
    # "flash-attn",
    "ninja",
    "xformers==0.0.29.post3",
]

RAG=[
    "pymilvus",
    "chromadb",
]

UI=[
    "stqdm",
    "ipywidgets",
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

IMAGES=[
    "labelme",
    "ultralytics"
    ]

setup(
    name="film69",
    version="0.4.9.dev0",
    author="Watcharaphon Pamayayang",
    author_email="filmmagic45@gmail.com",
    url="https://github.com/watcharaphon6912",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "tts=FILM69.tts.f5_tts.infer.infer_cli:main",
            "tts_train_ui=FILM69.tts.f5_tts.train.finetune_gradio:main",
        ],
    },
    extras_require={
        "llm": common_packages + LLM,
        "rag": common_packages + RAG,
        "speech": common_packages + SPEECH,
        "ui": UI,
        "iot":IOT,
        "images":IMAGES,
        "all": common_packages + ubuntu + LLM + RAG + UI + SPEECH + IOT + langchain + IMAGES,
        "all_win": common_packages + LLM + RAG + UI + SPEECH + IOT + langchain + IMAGES,
        "all_llama-cpp": common_packages + LLM + RAG + UI + SPEECH + langchain + ["llama-cpp-python"] + IMAGES
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    cmdclass={
        'install': InstallCommand,
    },
)
