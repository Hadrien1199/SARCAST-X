import gensim.downloader as api
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

word2vec_transfer = api.load("glove-wiki-gigaword-50")

# Function to convert a sentence (list of words) into a matrix representing the words in the embedding space
def embed_sentence_with_TF(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)

# Function that converts a list of sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

def pad_sequences_sarcasm(df_sarcasm):

    #split df_sarcasm into train and test using slicing
    train_data= df_sarcasm[:int(len(df_sarcasm)*0.8)]
    test_data= df_sarcasm[int(len(df_sarcasm)*0.8):]

    X_train = train_data['text']
    X_test = test_data['text']
    y_train = train_data['class']
    y_test = test_data['class']
    #Map NOT_SARCASM to 0 and SARCASM to 1
    y_train = y_train.map({'NOT_SARCASM': 0, 'SARCASM': 1})
    y_test = y_test.map({'NOT_SARCASM': 0, 'SARCASM': 1})
    # Embed the training and test sentences
    X_train_embed_2 = embedding(word2vec_transfer, X_train)
    X_test_embed_2 = embedding(word2vec_transfer, X_test)
    # Pad the training and test embedded sentences
    X_train_pad_2 = pad_sequences(X_train_embed_2, dtype='float32', padding='post', maxlen=200)
    X_test_pad_2 = pad_sequences(X_test_embed_2, dtype='float32', padding='post', maxlen=200)
    return X_train_pad_2, X_test_pad_2, y_train, y_test
