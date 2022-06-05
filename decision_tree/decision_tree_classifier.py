from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn import metrics
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

TEST_SIZE = 0.2


def predict_accuracy(labels, predicted_labels):
    correct_predictions = 0
    for i in range(len(labels)):
        if labels[i] == predicted_labels[i]:
            correct_predictions += 1
    
    return float(correct_predictions) / len(labels)


if __name__ == "__main__":
    print ("--------- DECISION TREE CLASSIFICATION using IRIS -------")
    data, labels = load_iris(return_X_y=True)
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

    print ("Train Shape: ", X_train.shape)
    print ("Test Shape: ", X_test.shape)

    DT_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
    DT_classifier.fit(X_train, y_train)

    # Print the decision tree
    tree_rules = export_text(DT_classifier)
    print ("\nDecision Tree:-\n")
    print(tree_rules)

    predicted_tree = DT_classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted_tree)
    print ("\nAccuracy: ", accuracy)
