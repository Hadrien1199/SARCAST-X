import pandas as pd
from DL_logic.params import *
from utils.data_preprocessing import preprocess_sarcasm_data, preprocess_fakenews_data
from colorama import Fore, Style

# load datasets
def load_social_sarcasm_data():
    df_GEN = pd.read_csv('raw_data/GEN-sarc-notsarc.csv')
    df_RQ = pd.read_csv('raw_data/RQ-sarc-notsarc.csv')
    df_HYP = pd.read_csv('raw_data/HYP-sarc-notsarc.csv')
    df_sarcasm = pd.concat([df_GEN, df_RQ, df_HYP])
    df_sarcasm['class'] = df_sarcasm['class'].apply(lambda x: 'SARCASM' if x == 'sarc' else 'NOT_SARCASM')
    #drop id column
    df_sarcasm.drop(columns=['id'], inplace=True)

    if not T5_MODEL_USED:
        #json files
        df_reddit_train=pd.read_json('raw_data/reddit_train.jsonl',lines=True)
        df_twitter_train = pd.read_json('raw_data/twitter_train.jsonl',lines=True)
        #merge reddit and twitter data
        df_social = pd.concat([df_reddit_train, df_twitter_train])
        #rename'label' to 'class'
        df_social.rename(columns={'label':'class'},inplace=True)
        df_social.rename(columns={'response':'text'},inplace=True)
        #dropcontext column
        df_social.drop(columns=['context'], inplace=True)
        # merge datasets
        df_social_sarcasm = pd.concat([df_sarcasm, df_social])
        # drop duplicates
        df_social_sarcasm.drop_duplicates(inplace=True)
        return df_social_sarcasm

    return df_sarcasm

def load_fakenews_data():
    df_fake = pd.read_csv('raw_data/twitter_sentiment_data.csv')
    df_fake = df_fake[df_fake['sentiment'] != 0]
    df_fake['sentiment']=df_fake['sentiment'].apply(lambda x: 'consensus' if x==1 or x==2 else 'non-consensus')
    return df_fake

def load_twitter_climate_data():
    df_climate = pd.read_csv('raw_data/twitter_climate.csv')
    return df_climate



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
    print(Fore.BLUE + "\nLoading & preprocessing fake news data..." + Style.RESET_ALL)
    df_fake = load_fakenews_data()
    # preprocess the text
    df_fake['message'] = df_fake['message'].apply(preprocess_fakenews_data)
    print("✅ Fake news data loaded & preprocessed")

    return df_sarcasm, df_fake
