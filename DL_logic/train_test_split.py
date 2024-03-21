
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from params import *
from warnings import filterwarnings
filterwarnings('ignore')
from colorama import Fore, Style

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


def train_test_fakenews(X, y, tokenizer_fake_news):

    print(Fore.BLUE + "\nTokenizing & splitting fake news data..." + Style.RESET_ALL)

    X_train_fake, X_test, y_train_fake, y_test_fake = train_test_split(X, y, test_size=0.2, random_state=42)
    #label encode the target variable
    le = LabelEncoder()
    y_train_fake = le.fit_transform(y_train_fake)
    y_test_fake = le.transform(y_test_fake)

    # tokenize the data
    train_encodings = tokenizer_fake_news(X_train_fake.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer_fake_news(X_test.tolist(), truncation=True, padding=True)

    # create a dataset
    class FakeNewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_fake_dataset = FakeNewsDataset(train_encodings, y_train_fake)
    test_fake_dataset = FakeNewsDataset(test_encodings, y_test_fake)
    print("✅ Fake news data tokenized & split")
    return train_fake_dataset, test_fake_dataset, y_test_fake, le
