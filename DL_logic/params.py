import os

# WORKING DIRECTORY: /DL_logic
WORKING_DIRECTORY = os.getcwd()

# HUGGINGFACE MODEL PATHS
HELINIVAN_MODEL_PATH = "helinivan/english-sarcasm-detector"
FAKE_NEWS_MODEL_PATH = "vikram71198/distilroberta-base-finetuned-fake-news-detection"
CLIMATEBERT_MODEL_PATH = "amandakonet/climatebert-fact-checking"
T5_SARCASM_MODEL_PATH = "mrm8488/t5-base-finetuned-sarcasm-twitter"
# SAVED MODELS
SARCASM_MODEL_SAVED = False
FAKE_NEWS_MODEL_SAVED = False

# Which model to use
T5_MODEL_USED = False
# CUDA
CUDA = False
