import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments
from models_huggingface import *
from train_test_split import *
from params import *
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style
if not CUDA:#DO NOT USE CUDA
    torch.device("cpu")


############################################ SARCASM MODEL ############################################

def train_sarcasm_model(model, train_dataset, test_dataset):
    """
    Train the sarcasm model.

    Args:
        model (object): The model object.
        train_dataset (object): The training dataset.
        test_dataset (object): The test dataset.

    Returns:
        object: The trained model.
    """
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
    trainer.train()
    print("✅ Sarcasm model trained")
    return trainer


def evaluate_sarcasm_model(trainer, test_dataset, y_test, le):
    """
    Evaluate the sarcasm model.

    Args:
        trainer (object): The trained model object.
        test_dataset (object): The test dataset.
        y_test (array): The true labels.
        le (object): The label encoder.

    Returns:
        float: The accuracy of the model.
    """
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
    """
    Save the sarcasm model.

    Args:
        model (object): The model object.
        tokenizer (object): The tokenizer object.
    """
    print(Fore.BLUE + "\nSaving sarcasm model..." + Style.RESET_ALL)
    # save the model
    model.save_pretrained("sarcasm_model")
    tokenizer.save_pretrained("sarcasm_model")
    print("✅ Sarcasm model saved")


def train_fake_news_model(model, train_dataset, test_dataset):
    """
    Train the fake news model.

    Args:
        model (object): The model object.
        train_dataset (object): The training dataset.
        test_dataset (object): The test dataset.

    Returns:
        object: The trained model.
    """
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
    # TODO: add the evaluation strategy, metrics, callbacks
    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # train the model
    trainer.train()
    print("✅ Fake news model trained")
    return trainer


def get_probabilities(text, fake_news_model, tokenizer_fake_news, sarcasm_model, tokenizer_sarcasm):
    """
    Get the probabilities of sarcasm and fake news for a given text.

    Args:
        text (str): The input text.
        fake_news_model (object): The fake news model object.
        tokenizer_fake_news (object): The fake news tokenizer object.
        sarcasm_model (object): The sarcasm model object.
        tokenizer_sarcasm (object): The sarcasm tokenizer object.

    Returns:
        tuple: The sarcasm probability and the fake news probability.
    """
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
