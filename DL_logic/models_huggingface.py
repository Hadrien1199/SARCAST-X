from params import *
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

def sarcasm_model_loader(sarcasm_model_path=SARCASM_MODEL_PATH):

    print(Fore.BLUE + "\nLoading sarcasm model..." + Style.RESET_ALL)
    # Load the model
    tokenizer_sarcasm = AutoTokenizer.from_pretrained(sarcasm_model_path)
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_path)
    print("✅ Sarcasm model loaded")


    return sarcasm_model, tokenizer_sarcasm

def fake_news_model_loader(model_path=FAKE_NEWS_MODEL_PATH):

    print(Fore.BLUE + "\nLoading fake news model..." + Style.RESET_ALL)
    # Load the model
    tokenizer_fake_news = AutoTokenizer.from_pretrained(model_path)
    fake_news_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("✅ Fake news model loaded")
    return fake_news_model, tokenizer_fake_news
