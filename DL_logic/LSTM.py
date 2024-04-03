from DL_logic.model_training_eval import *
from utils.load_data import load_preprocess_text

if __name__ == "__main__":

    # Load & preprocess the text
    df_sarcasm, df_fake = load_preprocess_text()

    if LSTM_SARCASM_USED:
            sarcasm_model= train_eval_LSTM_model_sarcasm(df_sarcasm)
            save_LSTM_sarcasm_model(sarcasm_model)
    if LSTM_FAKE_NEWS_USED:
            fake_news_model= train_eval_LSTM_model_fakenews(df_fake)
            save_LSTM_fakenews_model(fake_news_model)
