import string
#import stopwords
from nltk.corpus import stopwords

def preprocess_sarcasm_data(text):

    # Remove 'RT'
    text = text.replace('RT', '')

    #remove words that include 'emoticon'
    text = ' '.join([word for word in text.split() if 'emoticon' not in word])
    # remove whole words containing '@'
    text = ' '.join([word for word in text.split() if '@' not in word])
    # Remove urls
    text = ' '.join([word for word in text.split() if 'http' not in word])
    # Remove #
    text = text.replace('#', '')
    text = text.lower()
    # text = text.translate(str.maketrans('', '', string.punctuation))

    return text


def preprocess_fakenews_data(text):
    # Remove 'RT'
    text = text.replace('RT', '')

    #remove words that include 'emoticon'
    text = ' '.join([word for word in text.split() if 'emoticon' not in word])
    # remove whole words containing '@'
    text = ' '.join([word for word in text.split() if '@' not in word])
    # Remove urls
    text = ' '.join([word for word in text.split() if 'http' not in word])
    # Remove #
    text = text.replace('#', '')

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
