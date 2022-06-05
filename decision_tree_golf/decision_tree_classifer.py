import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier


TEST_SIZE = 0.2

FEATURES = {
    'Rainy': 0,
    'Overcast': 1,
    'Sunny': 2,
    'Hot': 0,
    'Mild': 1, 
    'Cool': 2,
    'High': 0,
    'Normal': 1,
    'False': 0,
    'True': 1,
    'No': 0,
    'Yes': 1
}


def load_dataset(filename):
    # use pandas to excel and convert to numpy array
    data = pd.read_excel(filename)
    data = np.array(data)
    print ("\nDataset shape: ", data.shape)
    print ("First row from the dataset:-\n", data[0])
    
    # get the last column as labels, remove 1st and last column from data.
    labels = [int(FEATURES[x[-1]]) for x in data]
    dataset = np.delete(data, (0, -1), axis=1)

    # Map strings to ints
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = FEATURES[str(dataset[i][j])]

    return dataset, labels


def predict_accuracy(labels, predicted_labels):
    correct_predictions = 0
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            correct_predictions += 1
    
    return float(correct_predictions) / len(labels)


if __name__ == "__main__":
    data, labels = load_dataset("golf_data.xlsx")    
    print ("\nFirst row of filtered and mapped dataset:-\n", data[0])

    print ("Dataset Shape: ", data.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, 
        test_size=TEST_SIZE, 
        random_state=42
    )

    # convert data for numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print ("\nTrain Shape: ", X_train.shape)
    print ("Test Shape: ", X_test.shape)

    print ("\n___CLASSIFICATION USING DECISION TREE___\n")
    DT_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
    DT_classifier.fit(X_train, y_train)

    # Print the decision tree
    tree_rules = export_text(DT_classifier)
    print ("\nDecision Tree:-\n")
    print(tree_rules)

    predicted_tree = DT_classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted_tree)
    print ("Accuracy: ", accuracy)

    print ("\n___CLASSIFICATION USING NAIVE BAYES___\n")
    guas_naive_bayes = GaussianNB()
    predicted_labels = guas_naive_bayes.fit(X_train, y_train).predict(X_test)

    accuracy = predict_accuracy(y_test, predicted_labels)

    print ("Accuracy: ", accuracy)
