import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import evaluate
from transformers import Trainer, TrainingArguments
from models_huggingface import *
from train_test_split import *
from params import *
from utils.word2vec import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style
if not CUDA:#DO NOT USE CUDA
    torch.device("cpu")

############################################ METRIC Function ############################################
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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
        output_dir='./results/sarcasm',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch"
    )

    # TODO: add the evaluation strategy, metrics, callbacks
    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
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
    print(Fore.MAGENTA +"\nClassification Report" + Style.RESET_ALL)
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
    model.save_pretrained("models/BERT_models/sarcasm_model")
    tokenizer.save_pretrained("models/BERT_models/sarcasm_model")
    print("✅ Sarcasm model saved")


############################################ FAKE NEWS MODEL ############################################
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
        output_dir='./results/fake_news',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch"
    )
    # TODO: add the evaluation strategy, metrics, callbacks
    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # train the model
    trainer.train()
    print("✅ Fake news model trained")
    return trainer

def evaluate_fake_news_model(trainer, test_dataset, y_test, le):

        """
        Evaluate the fake news model.

        Args:
            trainer (object): The trained model object.
            test_dataset (object): The test dataset.
            y_test (array): The true labels.
            le (object): The label encoder.

        Returns:
            float: The accuracy of the model.
        """

        print(Fore.BLUE + "\nEvaluating fake news model..." + Style.RESET_ALL)
        # evaluate the model
        trainer.evaluate()

        # make predictions
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(-1)

        # calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        # print the classification report
        print(Fore.MAGENTA +"\nClassification Report" + Style.RESET_ALL)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        print("✅ Fake news model evaluated")
        return accuracy

def save_fake_news_model(model, tokenizer):

        """
        Save the fake news model.

        Args:
            model (object): The model object.
            tokenizer (object): The tokenizer object.
        """

        print(Fore.BLUE + "\nSaving fake news model..." + Style.RESET_ALL)
        # save the model
        model.save_pretrained("models/BERT_models/fake_news_model")
        tokenizer.save_pretrained("models/BERT_models/fake_news_model")
        print("✅ Fake news model saved")



############################################ LSTM SARCASM MODEL ############################################

def train_eval_LSTM_model_sarcasm(df_sarcasm):
    """
    Train the LSTM model.

    Args:
        df (Dataframe): The dataframe containing the data.

    Returns:
        object: The trained model.
    """

    print(Fore.BLUE + "\nEmbedding & padding data..." + Style.RESET_ALL)
    X_train, X_test, y_train, y_test = pad_sequences_sarcasm(df_sarcasm)
    print("✅ Data embedded & padded")


    parameters = {
                'epochs': 100,
                'learning_rate': 0.001,
                'decay': 0.1,
                'batch_size': 32,
                'metrics': ['accuracy'],
                'patience': 10,
                'monitor': 'val_loss',
                'min_delta': 0.01
                  }

    print(Fore.BLUE + "\nIntializing & compiling LSTM model..." + Style.RESET_ALL)
    model = Sequential()
    model.add(layers.Masking())
    model.add(layers.LSTM(20, activation='tanh'))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=parameters['metrics'])
    print("✅ LSTM model initialized & compiled")

    early_stopping = EarlyStopping(
                                    monitor=parameters['monitor'],
                                    patience=parameters['patience'],
                                    min_delta=parameters['min_delta']
                                    )

    print(Fore.BLUE + "\nTraining LSTM model..." + Style.RESET_ALL)
    model.fit(
            X_train, y_train,
            epochs=parameters['epochs'],
            batch_size=parameters['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping]
            )
    print("✅ LSTM model trained")

    print(Fore.BLUE + "\nEvaluating LSTM model..." + Style.RESET_ALL)
    y_pred = model.predict(X_test)
    # classification report
    y_pred = (y_pred > 0.5)
    test_performance = model.evaluate(X_test, y_test)
    print('Test performance: ', test_performance)
    print(Fore.MAGENTA +"\nClassification Report" + Style.RESET_ALL)
    print(classification_report(y_test, y_pred))
    print("✅ LSTM model evaluated")
    return model

def save_LSTM_sarcasm_model(model):
    """
    Save the LSTM model.

    Args:
        model (object): The model object.
    """

    print(Fore.BLUE + "\nSaving LSTM model..." + Style.RESET_ALL)
    # save the model
    model.save("models/LSTM_models/sarcasm_model/V1.h5")
    print("✅ LSTM model saved")
    return model


############################################ LSTM FAKE NEWS MODEL ############################################

def train_eval_LSTM_model_fakenews(df_fake):
    """
    Train the LSTM model.

    Args:
        df (Dataframe): The dataframe containing the data.

    Returns:
        object: The trained model.
    """

    print(Fore.BLUE + "\nEmbedding & padding data..." + Style.RESET_ALL)
    X_train, X_test, y_train, y_test = pad_sequences_fakenews(df_fake)
    print("✅ Data embedded & padded")

    parameters = {
                'epochs': 100,
                'learning_rate': 0.001,
                'decay': 0.1,
                'batch_size': 32,
                'metrics': ['accuracy'],
                'patience': 10,
                'monitor': 'val_accuracy',
                'min_delta': 0.01
                  }

    print(Fore.BLUE + "\nIntializing & compiling LSTM model..." + Style.RESET_ALL)
    model = Sequential()
    model.add(layers.Masking())
    model.add(layers.LSTM(20, activation='tanh'))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=parameters['metrics'])
    print("✅ LSTM model initialized & compiled")

    early_stopping = EarlyStopping(
                                    monitor=parameters['monitor'],
                                    patience=parameters['patience'],
                                    min_delta=parameters['min_delta']
                                    )

    print(Fore.BLUE + "\nTraining LSTM model..." + Style.RESET_ALL)
    model.fit(
            X_train, y_train,
            epochs=parameters['epochs'],
            batch_size=parameters['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping]
            )
    print("✅ LSTM model trained")

    print(Fore.BLUE + "\nEvaluating LSTM model..." + Style.RESET_ALL)
    y_pred = model.predict(X_test)
    # classification report
    y_pred = (y_pred > 0.5)
    test_performance = model.evaluate(X_test, y_test)
    print('Test performance: ', test_performance)
    print(Fore.MAGENTA +"\nClassification Report" + Style.RESET_ALL)
    print(classification_report(y_test, y_pred))
    print("✅ LSTM model evaluated")

    return model

def save_LSTM_fakenews_model(model):
    """
    Save the LSTM model.

    Args:
        model (object): The model object.
    """

    print(Fore.BLUE + "\nSaving LSTM model..." + Style.RESET_ALL)
    # save the model
    model.save("models/LSTM_models/fake_news_model/V1.h5")
    print("✅ LSTM model saved")
    return model
