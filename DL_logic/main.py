import torch
from utils.data_cleaning import preprocess_sarcasm_data, preprocess_fakenews_data
from utils.load_data import load_social_sarcasm_data, load_fakenews_data
from models_huggingface import *
from model_functions import *
from train_test_split import *
from params import *
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style
import time
from tqdm import tqdm
if not CUDA:#DO NOT USE CUDA
    torch.device("cpu")


############################################ LOAD & PREPROCESS TEXT ############################################

def load_preprocess_text():
    """
    Load and preprocess the sarcasm and fake news data.

    Returns:
        tuple: A tuple containing two pandas DataFrames - df_sarcasm and df_fake.
               df_sarcasm: DataFrame containing the preprocessed sarcasm data.
               df_fake: DataFrame containing the preprocessed fake news data.
    """
    # Load the data
    print(Fore.BLUE + "\nLoading & preprocessing sarcasm data..." + Style.RESET_ALL)
    df_sarcasm = load_social_sarcasm_data()
    # preprocess the text
    df_sarcasm['text'] = df_sarcasm['text'].apply(preprocess_sarcasm_data)
    print("✅ Sarcasm data loaded & preprocessed")

    # load the fake news data
    print(Fore.BLUE + "\nLoading fake news data..." + Style.RESET_ALL)
    df_fake = load_fakenews_data()
    # preprocess the text
    df_fake['message'] = df_fake['message'].apply(preprocess_fakenews_data)
    print("✅ Fake news data loaded & preprocessed")

    return df_sarcasm, df_fake


############################################ MAIN FUNCTION ############################################

if __name__ == "__main__":

    # Load & preprocess the text
    df_sarcasm, df_fake = load_preprocess_text()

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

    # Retrain the fake news model
    # fake_news_model, tokenizer_fake_news = fake_news_model_loader()

    # Get the probabilities

    # Sarcasm
    print(Fore.BLUE + "\nLoading fine-tuned sarcasm model..." + Style.RESET_ALL)
    # load the sarcasm model
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained("models/sarcasm_model")
    tokenizer_sarcasm = AutoTokenizer.from_pretrained("models/sarcasm_model")
    print("✅ Model loaded")

    print(Fore.BLUE + "\nComputing sarcasm probabilities..." + Style.RESET_ALL)
    with tqdm(total=len(df_fake)) as pbar:
        df_fake['fake_news_prob'] = df_fake['message'].apply(lambda x: get_sarcasm_probabilities(x,tokenizer_sarcasm,sarcasm_model))
        pbar.update(1)
    print("✅ Sarcasm probabilities computed")

    # Fake news
    print(Fore.BLUE + "\nLoading fine-tuned fake-news model..." + Style.RESET_ALL)
    # Load the fake news model
    tokenizer_fake_news = AutoTokenizer.from_pretrained("vikram71198/distilroberta-base-finetuned-fake-news-detection")
    fake_news_model = AutoModelForSequenceClassification.from_pretrained("vikram71198/distilroberta-base-finetuned-fake-news-detection")
    print("✅ Model loaded")

    print(Fore.BLUE + "\nComputing fake news probabilities..." + Style.RESET_ALL)
    with tqdm(total=len(df_fake)) as pbar:
        df_fake['sarcasm_prob'] = df_fake['message'].apply(lambda x: get_fakenews_probabilities(x,tokenizer_fake_news,fake_news_model))
        pbar.update(1)
    print("✅ Fake news probabilities computed")

    # Final dataframe
    print(Fore.MAGENTA + "\nFinal dataframe:" + Style.RESET_ALL)
    print(df_fake.head())
