# Load iris dataset from sklearn datsets and use Naive bayes to classify

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np

TEST_SIZE = 0.2


def predict_accuracy(labels, predicted_labels):
    correct_predictions = 0
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            correct_predictions += 1
    
    return float(correct_predictions) / len(labels)


if __name__ == "__main__":
    print ("--------- NAIVE BAYES USING IRIS -------")
    data, labels = load_iris(return_X_y=True)
    print ("Dataset Shape: ", data.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, 
        test_size=TEST_SIZE, 
        random_state=0
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

    

