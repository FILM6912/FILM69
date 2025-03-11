import warnings
warnings.simplefilter("ignore", UserWarning)

try:from .ml import *
except:...
from .tts import TTS
from .datasets.clean_text import clean_text
