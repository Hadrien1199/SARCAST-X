from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import string
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments
from utils.data_cleaning import preprocess_sarcasm_data, preprocess_fakenews_data
from utils.load_data import load_data, load_fakenews_data
from params import *
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style

def load_preprocess_text():

    # Load the data
    print(Fore.BLUE + "\nLoading & preprocessing sarcasm data..." + Style.RESET_ALL)
    df_sarcasm = load_data()
    # preprocess the text
    df_sarcasm['text'] = df_sarcasm['text'].apply(preprocess_sarcasm_data)
    print("✅ Sarcasm data loaded & preprocessed")

    # load the fake news data
    print(Fore.BLUE + "\nLoading fake news data..." + Style.RESET_ALL)
    df_fake = load_fakenews_data()
    # preprocess the text
    df_fake['text'] = df_fake['text'].apply(preprocess_fakenews_data)

    return df_sarcasm, df_fake


############################################ SARCASM MODEL ############################################

def sarcasm_model_loader(sarcasm_model_path=SARCASM_MODEL_PATH):

    print(Fore.BLUE + "\nLoading sarcasm model..." + Style.RESET_ALL)
    # Load the model
    tokenizer_sarcasm = AutoTokenizer.from_pretrained(sarcasm_model_path)
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_path)
    print("✅ Sarcasm model loaded")


    return sarcasm_model, tokenizer_sarcasm

def train_test_sarcasm(X, y, tokenizer_sarcasm):

    print(Fore.BLUE + "\nTokenizing & splitting sarcasm data..." + Style.RESET_ALL)

    X_train_sarcasm, X_test, y_train_sarcasm, y_test_sarcasm = train_test_split(X, y, test_size=0.2, random_state=42)
    #label encode the target variable
    le = LabelEncoder()
    y_train_sarcasm = le.fit_transform(y_train_sarcasm)
    y_test_sarcasm = le.transform(y_test_sarcasm)

    # tokenize the data
    train_encodings = tokenizer_sarcasm(X_train_sarcasm.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer_sarcasm(X_test.tolist(), truncation=True, padding=True)

    # create a dataset
    class SarcasmDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_sarcasm_dataset = SarcasmDataset(train_encodings, y_train_sarcasm)
    test_sarcasm_dataset = SarcasmDataset(test_encodings, y_test_sarcasm)
    print("✅ Sarcasm data tokenized & split")
    return train_sarcasm_dataset, test_sarcasm_dataset, y_test_sarcasm, le

def train_sarcasm_model(model, train_dataset, test_dataset):

    print(Fore.BLUE + "\nTraining sarcasm model..." + Style.RESET_ALL)
    # define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # train the model
    trainer = trainer.train()
    print("✅ Sarcasm model trained")
    return trainer


def evaluate_sarcasm_model(trainer, test_dataset, y_test, le):

    print(Fore.BLUE + "\nEvaluating sarcasm model..." + Style.RESET_ALL)
    # evaluate the model
    trainer.evaluate()

    # make predictions
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    # print the classification report
    print(Fore.MAGENTA +"\nClassification Report")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("✅ Sarcasm model evaluated")
    return accuracy

def save_sarcasm_model(model, tokenizer):

    print(Fore.BLUE + "\nSaving sarcasm model..." + Style.RESET_ALL)
    # save the model
    model.save_pretrained("sarcasm_model")
    tokenizer.save_pretrained("sarcasm_model")
    print("✅ Sarcasm model saved")
    return

############################################ FAKE NEWS MODEL ############################################

def fake_news_model_loader(model_path=FAKE_NEWS_MODEL_PATH):

    print(Fore.BLUE + "\nLoading fake news model..." + Style.RESET_ALL)
    # Load the model
    tokenizer_fake_news = AutoTokenizer.from_pretrained(model_path)
    fake_news_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("✅ Fake news model loaded")
    return fake_news_model, tokenizer_fake_news

############################################ GET PROBABILITIES ############################################

def get_probabilities(text, fake_news_model, tokenizer_fake_news, sarcasm_model, tokenizer_sarcasm):

    print(Fore.BLUE + "\Computing fake news probabilities..." + Style.RESET_ALL)
    # get the probabilities of the fake news model
    fake_news_input = tokenizer_fake_news(text, truncation = True, padding = "max_length", max_length = 512, return_tensors='pt')
    fake_news_output = fake_news_model(**fake_news_input)
    fake_news_prob = torch.nn.functional.softmax(fake_news_output.logits, dim=-1)
    fake_news_prob = fake_news_prob.detach().numpy()
    fake_news_prob = fake_news_prob[0][1]
    print("✅ Fake news probabilities computed")

    print(Fore.BLUE + "\nComputing sarcasm probabilities..." + Style.RESET_ALL)
    # get the probabilities of the fake news model
    sarcasm_input = tokenizer_sarcasm(text, return_tensors="pt", padding=True, truncation=True)
    sarcasm_output = sarcasm_model(**sarcasm_input)
    sarcasm_prob = torch.nn.functional.softmax(sarcasm_output.logits, dim=-1)
    sarcasm_prob = sarcasm_prob.detach().numpy()
    sarcasm_prob = sarcasm_prob[0][1]
    print("✅ Sarcasm probabilities computed")

    return sarcasm_prob, fake_news_prob

if __name__ == "__main__":

    # Retrain the sarcasm model
    df_sarcasm, df_fake = load_preprocess_text()
    sarcasm_model, tokenizer_sarcasm = sarcasm_model_loader()
    X = df_sarcasm['text']
    y = df_sarcasm['class']
    train_dataset, test_dataset, y_test, le = train_test_sarcasm(X, y, tokenizer_sarcasm)
    trainer = train_sarcasm_model(sarcasm_model, train_dataset, test_dataset)
    evaluate_sarcasm_model(trainer, test_dataset, y_test, le)
    save_sarcasm_model(sarcasm_model, tokenizer_sarcasm)
    # Load the fake news model
    fake_news_model, tokenizer_fake_news = fake_news_model_loader()
    # Get the probabilities
    df_fake['sarcasm_prob'] = df_fake['text'].apply(lambda x: get_probabilities(x, fake_news_model, tokenizer_fake_news, sarcasm_model, tokenizer_sarcasm)[0])
    df_fake['fake_news_prob'] = df_fake['text'].apply(lambda x: get_probabilities(x, fake_news_model, tokenizer_fake_news, sarcasm_model, tokenizer_sarcasm)[1])
    print(df_fake.head())
