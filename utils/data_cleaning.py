import string

def preprocess_sarcasm_data(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    #remove words that include 'emoticon'
    text = ' '.join([word for word in text.split() if 'emoticon' not in word])
    return text


def preprocess_fakenews_data(text):
    # Remove 'RT'
    text = text.replace('RT', '')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    #remove words that include 'emoticon'
    text = ' '.join([word for word in text.split() if 'emoticon' not in word])
    # remove whole words containing '@'
    text = ' '.join([word for word in text.split() if '@' not in word])
    # Remove urls
    text = ' '.join([word for word in text.split() if 'http' not in word])
    # Remove #
    text = text.replace('#', '')

    return text
