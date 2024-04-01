import torch
from models_huggingface import *
from train_test_split import *
from params import *
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style
if not CUDA:#DO NOT USE CUDA
    torch.device("cpu")




############################################ GET PROBABILITIES ############################################

def get_fakenews_probabilities(text,tokenizer_fake_news,fake_news_model):

    """
    Get the probabilities of sarcasm and fake news for a given text.

    Args:
        text (str): The input text.

    Returns:
        float: The fake news probability.
    """

    # get the probabilities of the fake news model
    fake_news_input = tokenizer_fake_news(text, truncation = True, padding = "max_length", max_length = 512, return_tensors='pt')
    fake_news_output = fake_news_model(**fake_news_input)
    fake_news_prob = torch.nn.functional.softmax(fake_news_output.logits, dim=-1)
    fake_news_prob = fake_news_prob.detach().numpy()
    fake_news_prob = fake_news_prob[0][1]

    return fake_news_prob


def get_sarcasm_probabilities(text,tokenizer_sarcasm,sarcasm_model):

    """
    Get the probabilities of sarcasm and fake news for a given text.

    Args:
        text (str): The input text.

    Returns:
        float: The sarcasm probability.
    """

    # get the probabilities of the fake news model
    sarcasm_input = tokenizer_sarcasm(text, return_tensors="pt", padding=True, truncation=True)
    sarcasm_output = sarcasm_model(**sarcasm_input)
    sarcasm_prob = torch.nn.functional.softmax(sarcasm_output.logits, dim=-1)
    sarcasm_prob = sarcasm_prob.detach().numpy()
    sarcasm_prob = sarcasm_prob[0][1]

    return sarcasm_prob


# Get the probabilities
def get_probs(df_fake):
    """
    Get the probabilities of sarcasm and fake news for the given text.

    Returns:
        DataFrame: A DataFrame containing the probabilities of sarcasm and fake news.
    """

    # Sarcasm
    print(Fore.BLUE + "\nLoading fine-tuned sarcasm model..." + Style.RESET_ALL)
    # load the sarcasm model
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained("models/sarcasm_model")
    tokenizer_sarcasm = AutoTokenizer.from_pretrained("models/sarcasm_model")
    print("✅ Model loaded")

    print(Fore.BLUE + "\nComputing sarcasm probabilities..." + Style.RESET_ALL)
    with tqdm(total=len(df_fake)) as pbar:
        df_fake['sarcasm_prob'] = df_fake['message'].apply(lambda x: get_sarcasm_probabilities(x,tokenizer_sarcasm,sarcasm_model))
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
        df_fake['fake_news_prob'] = df_fake['message'].apply(lambda x: get_fakenews_probabilities(x,tokenizer_fake_news,fake_news_model))
        pbar.update(1)
    print("✅ Fake news probabilities computed")
    return df_fake
