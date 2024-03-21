import pandas as pd

# load datasets
def load_data():
    df_GEN = pd.read_csv('raw_data/GEN-sarc-notsarc.csv')
    df_RQ = pd.read_csv('raw_data/RQ-sarc-notsarc.csv')
    df_HYP = pd.read_csv('raw_data/HYP-sarc-notsarc.csv')
    # merge datasets
    df_sarcasm = pd.concat([df_GEN, df_RQ, df_HYP])
    return df_sarcasm

def load_fakenews_data():
    df_fake = pd.read_csv('raw_data/twitter_sentiment_data.csv')
    return df_fake

def load_twitter_climate_data():
    df_climate = pd.read_csv('raw_data/twitter_climate.csv')
    return df_climate
