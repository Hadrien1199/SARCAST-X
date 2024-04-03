import torch
from utils.load_data import load_preprocess_text
from models_huggingface import *
from DL_logic.model_training_eval import *
from train_test_split import *
from params import *
from get_probs_huggingface import *
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style
import time
from tqdm import tqdm
if not CUDA:#DO NOT USE CUDA
    torch.device("cpu")


############################################ MAIN FUNCTION ############################################

if __name__ == "__main__":

    # Load & preprocess the text
    df_sarcasm, df_fake = load_preprocess_text()

    if LSTM_MODEL_USED:
        sarcasm_model= train_eval_LSTM_model_sarcasm(df_sarcasm)

    if not SARCASM_MODEL_SAVED:
        # Retrain the sarcasm model
        sarcasm_model, tokenizer_sarcasm = sarcasm_model_loader()
        X_sarcasm = df_sarcasm['text']
        y_sarcasm = df_sarcasm['class']
        train_dataset, test_dataset, y_test, le = train_test_sarcasm(X_sarcasm, y_sarcasm, tokenizer_sarcasm)

        start_time = time.time()
        trainer = train_sarcasm_model(sarcasm_model, train_dataset, test_dataset)
        end_time = time.time()
        print(Fore.GREEN + f"\nTraining time: {end_time - start_time} seconds")

        evaluate_sarcasm_model(trainer, test_dataset, y_test, le)
        save_sarcasm_model(sarcasm_model, tokenizer_sarcasm)

    if not FAKE_NEWS_MODEL_SAVED:
        # Retrain the fake news model
        fake_news_model, tokenizer_fake_news = fake_news_model_loader()
        X_fake_news = df_fake['message']
        y_fake_news = df_fake['sentiment']
        train_dataset, test_dataset, y_test, le = train_test_fakenews(X_fake_news, y_fake_news, tokenizer_fake_news)

        start_time = time.time()
        trainer = train_fake_news_model(fake_news_model, train_dataset, test_dataset)
        end_time = time.time()
        print(Fore.GREEN + f"\nTraining time: {end_time - start_time} seconds")

        evaluate_fake_news_model(trainer, test_dataset, y_test, le)
        save_fake_news_model(fake_news_model, tokenizer_fake_news)

    # Get the probabilities

    df_fake= get_probs(df_fake)

    # # Final dataframe
    print(Fore.MAGENTA + "\nFinal dataframe:" + Style.RESET_ALL)
    print(df_fake.head())
