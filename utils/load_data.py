import pandas as pd

# load datasets
def load_social_sarcasm_data():
    df_GEN = pd.read_csv('raw_data/GEN-sarc-notsarc.csv')
    df_RQ = pd.read_csv('raw_data/RQ-sarc-notsarc.csv')
    df_HYP = pd.read_csv('raw_data/HYP-sarc-notsarc.csv')
    df_sarcasm = pd.concat([df_GEN, df_RQ, df_HYP])
    df_sarcasm['class']=df_sarcasm['class'].apply(lambda x: 1 if x=='sarc' else 0)
    #drop id column
    df_sarcasm.drop(columns=['id'], inplace=True)

    #json files
    df_reddit_train=pd.read_json('raw_data/reddit_train.jsonl',lines=True)
    df_twitter_train = pd.read_json('raw_data/twitter_train.jsonl',lines=True)
    #merge reddit and twitter data
    df_social = pd.concat([df_reddit_train, df_twitter_train])
    df_social['label']=df_social['label'].apply(lambda x: 1 if x=='SARCASM' else 0)
    #rename'label' to 'class'
    df_social.rename(columns={'label':'class'},inplace=True)
    df_social.rename(columns={'response':'text'},inplace=True)
    #dropcontext column
    df_social.drop(columns=['context'], inplace=True)

    # merge datasets
    df_social_sarcasm = pd.concat([df_sarcasm, df_social])

    return df_social_sarcasm

def load_fakenews_data():
    df_fake = pd.read_csv('raw_data/twitter_sentiment_data.csv')
    return df_fake

def load_twitter_climate_data():
    df_climate = pd.read_csv('raw_data/twitter_climate.csv')
    return df_climate
