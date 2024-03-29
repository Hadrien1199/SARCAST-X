import os

# WORKING DIRECTORY: /DL_logic
WORKING_DIRECTORY = os.getcwd()

# HUGGINGFACE MODEL PATHS
SARCASM_MODEL_PATH = "helinivan/english-sarcasm-detector"
FAKE_NEWS_MODEL_PATH = "vikram71198/distilroberta-base-finetuned-fake-news-detection"
CLIMATEBERT_MODEL_PATH = "amandakonet/climatebert-fact-checking"

# SAVED MODELS
SARCASM_MODEL_SAVED = True
FAKE_NEWS_MODEL_SAVED = False

# CUDA
CUDA = False
