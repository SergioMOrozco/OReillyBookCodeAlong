## Exercise 12 : 
# Implement Batch Gradient Descent with early stopping
# for Softmax Regression (without using Scitkit-Learn).

from sklearn import datasets
import soft_max_scratch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def convert_to_one_hot(labels):
    class_count = len(set(labels)) 
    one_hot = np.zeros((len(labels),class_count))
    
    for i in range(len(labels)):
        one_hot[i][labels[i]] = 1

    return one_hot

def main():
    iris = datasets.load_iris()

    data = iris["data"]
    labels_one_hot = convert_to_one_hot(iris["target"])
    
    rand = np.random.permutation(len(data))
    X_train, X_test, y_train, y_test = train_test_split(data[rand],
                                                        labels_one_hot[rand],
                                                        test_size = .33)

    soft_clf = soft_max_scratch.SoftMaxClassifier(learning_rate = 0.05,
                                                max_iter = 40000,
                                                minimum_step_size = 0.0001,
                                                early_stopping = True,
                                                n = 2000)
    soft_clf.train(X_train,y_train)


    y_pred = soft_clf.predict(X_test)

    print(accuracy_score(y_test,y_pred))

if __name__ == "__main__":
    main()


