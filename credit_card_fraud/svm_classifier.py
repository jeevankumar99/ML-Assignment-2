# Classifies whether a credit card transaction is fraud or not.

from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix

# 30000 datapoints total
TRAIN_SPLIT = 15000

def load_arff_dataset(filename):
    dataset = arff.loadarff(filename)
    data = pd.DataFrame(dataset[0])
    print (data)

def load_dataset(filename):
    # each value is split by a comma, so we use that as delimiter.
    data = pd.read_csv(filename)
    dataset = pd.DataFrame(data)

    return dataset


def get_train_data(dataset):
    dataset_train = dataset[0:TRAIN_SPLIT]
    dataset_train_fraud = dataset_train[dataset_train['Class'] == 1]
    dataset_train_no_fraud = dataset_train[dataset_train['Class'] == 0]

    dataset_fraud_sample = dataset_train_no_fraud.sample(300)
    dataset_balanced_train = dataset_train_fraud.append(dataset_fraud_sample)
    dataset__balanced_train = dataset_balanced_train.sample(frac=1)

    X_train = dataset_balanced_train.drop(['Time', 'Class'], axis=1)
    y_train = dataset_balanced_train['Class']

    return np.array(X_train), np.array(y_train)


def get_test_data(dataset):
    dataset_test = dataset[TRAIN_SPLIT:]
    X_test = dataset_test.drop(['Time', 'Class'], axis=1)
    y_test = dataset_test['Class']

    return np.array(X_test), np.array(y_test)
    

if __name__ == "__main__":
    dataset = load_dataset("credit_card_data.csv")

    print ("Dataset:-\n", dataset)

    X_train, y_train = get_train_data(dataset)

    X_test, y_test = get_test_data(dataset)

    print ("\nTraining....")
    
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit (X_train, y_train)

    print ("Training complete!")

    predicted_labels = svm_classifier.predict(X_test)

    # Construct confusion matrix and get accuracy
    confusion_mat = confusion_matrix(y_test, predicted_labels)
    accuracy = (confusion_mat[0][0] + confusion_mat[1][1]) / (
        sum(confusion_mat[0]) + sum(confusion_mat[1])
    )

    print("\nAccuracy: ", accuracy)