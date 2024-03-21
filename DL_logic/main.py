import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments
from utils.data_cleaning import preprocess_sarcasm_data, preprocess_fakenews_data
from utils.load_data import load_data, load_fakenews_data
from models_huggingface import *
from train_test_split import *
from params import *
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style



############################################ LOAD & PREPROCESS TEXT ############################################

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
    df_fake['message'] = df_fake['message'].apply(preprocess_fakenews_data)
    print("✅ Fake news data loaded & preprocessed")

    return df_sarcasm, df_fake


############################################ SARCASM MODEL ############################################

def train_sarcasm_model(model, train_dataset, test_dataset):

    print(Fore.BLUE + "\nTraining sarcasm model..." + Style.RESET_ALL)
    # define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
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

def train_fake_news_model(model, train_dataset, test_dataset):

    print(Fore.BLUE + "\nTraining fake news model..." + Style.RESET_ALL)
    # define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
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
    print("✅ Fake news model trained")
    return trainer



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


############################################ MAIN FUNCTION ############################################

if __name__ == "__main__":

    # Load & preprocess the text
    df_sarcasm, df_fake = load_preprocess_text()

    # Retrain the sarcasm model
    sarcasm_model, tokenizer_sarcasm = sarcasm_model_loader()
    X_sarcasm = df_sarcasm['text']
    y_sarcasm = df_sarcasm['class']
    train_dataset, test_dataset, y_test, le = train_test_sarcasm(X_sarcasm, y_sarcasm, tokenizer_sarcasm)
    trainer = train_sarcasm_model(sarcasm_model, train_dataset, test_dataset)
    evaluate_sarcasm_model(trainer, test_dataset, y_test, le)
    save_sarcasm_model(sarcasm_model, tokenizer_sarcasm)

    # Retrain the fake news model
    fake_news_model, tokenizer_fake_news = fake_news_model_loader()



    # Get the probabilities
    df_fake['sarcasm_prob'] = df_fake['message'].apply(lambda x: get_probabilities(x, fake_news_model, tokenizer_fake_news, sarcasm_model, tokenizer_sarcasm)[0])
    df_fake['fake_news_prob'] = df_fake['message'].apply(lambda x: get_probabilities(x, fake_news_model, tokenizer_fake_news, sarcasm_model, tokenizer_sarcasm)[1])
    print(df_fake.head())
