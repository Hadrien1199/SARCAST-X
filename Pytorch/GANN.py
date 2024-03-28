# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import random
from DL_logic.params import *


# Pytorch Generative Adversarial Networks for sarcasm detection

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, device):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embeds = self.word_embeddings(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, device):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        embeds = self.word_embeddings(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))


class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long)
        }

def train_gan(generator, discriminator, train_loader, criterion, optimizer_gen, optimizer_disc, num_epochs, device):
    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            real_data = data['input_ids'].to(device)
            real_data = real_data.view(-1, real_data.size(1))

            real_labels = torch.ones(real_data.size(0)).to(device)
            fake_labels = torch.zeros(real_data.size(0)).to(device)

            # Train the generator
            optimizer_gen.zero_grad()
            hidden_gen = generator.init_hidden(real_data.size(0))
            fake_data, _ = generator(real_data, hidden_gen)
            outputs, _ = discriminator(fake_data, discriminator.init_hidden(real_data.size(0)))
            gen_loss = criterion(outputs.view(-1), real_labels)
            gen_loss.backward()
            optimizer_gen.step()

            # Train the discriminator
            optimizer_disc.zero_grad()
            hidden_disc = discriminator.init_hidden(real_data.size(0))
            outputs_real, _ = discriminator(real_data, hidden_disc)
            real_loss = criterion(outputs_real.view(-1), real_labels)

            outputs_fake, _ = discriminator(fake_data.detach(), hidden_disc)
            fake_loss = criterion(outputs_fake.view(-1), fake_labels)

            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            optimizer_disc.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {disc_loss.item():.4f}')

def generate_text(generator, tokenizer, device, seed_text, max_length):
    generator.eval()
    with torch.no_grad():
        hidden = generator.init_hidden(1)
        seed_text = tokenizer.encode(seed_text, return_tensors='pt').to(device)
        seed_text = seed_text.view(-1, seed_text.size(1))
        output, _ = generator(seed_text, hidden)
        output = output.view(-1, output.size(1))
        output = torch.argmax(output, dim=1)
        output = output.tolist()
        text = tokenizer.decode(output)
        return text
