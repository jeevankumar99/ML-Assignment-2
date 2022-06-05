import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


TEST_SIZE = 0.2

def filter_dataset(data):
    re_space = re.compile('[/(){}\[\]\|@,;]')
    re_symbols = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    filtered_corpus = []

    for text in data:
        # lowercase text, replace space and symbols
        text = text.lower()
        text = re_space.sub(' ', text) 
        text = re_symbols.sub('', text)

        text = text.replace('x', '')
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)
        filtered_corpus.append(text)

    return np.array(filtered_corpus)


def load_dataset(filename):
    data = pd.read_csv(filename)
    
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data = data.drop_duplicates(keep="first")
    
    print("Dataset Shape: ", data['text'].shape)
    print ("i.e, 4993 mails of spam and ham classes!")

    return data['text'], data['label_num']


def predict_accuracy(labels, predicted_labels):
    correct_predictions = 0
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            correct_predictions += 1
    
    return float(correct_predictions) / len(labels)


def tokenize_data(data):
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer(max_features=2000)
    tokenized_data = tfidf_vectorizer.fit_transform(data).toarray()

    return tokenized_data


if __name__ == "__main__":
    print ("------ SPAM CLASSIFICATION using NAIVE BAYES ------")
    data, labels = load_dataset("scrapped_mail_data.csv")
    print("First 10 samples:-\n", data[:10])
    
    filtered_data = filter_dataset(data)     
    print ("Filtered Data:-\n", filtered_data[:10])
    
    tokenized_data = tokenize_data(filtered_data)
    
    X_train,X_test,y_train,y_test = train_test_split(
        tokenized_data, labels,
        test_size=0.2 ,random_state=2
        )

    # convert data for numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print ("Train Shape: ", X_train.shape)
    print ("Test Shape: ", X_test.shape)

    guas_naive_bayes = GaussianNB()
    predicted_labels = guas_naive_bayes.fit(X_train, y_train).predict(X_test)

    accuracy = predict_accuracy(y_test, predicted_labels)

    print ("\nAccuracy: ", accuracy)
