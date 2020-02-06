## Exercise 12 : 
# Implement Batch Gradient Descent with early stopping
# for Softmax Regression (without using Scitkit-Learn).

from sklearn import datasets
import soft_max_scratch
import numpy as np
iris = datasets.load_iris()

x = iris["data"]
y = iris["target"]

##soft_clf = soft_max_scratch.SoftMaxClassifier()
##soft_clf.train(x,[[1,0,0]])
##print(soft_clf.predict(x))


def convert_to_one_hot(labels):
    class_count = len(set(labels)) 
    one_hot = np.zeros((len(labels),class_count))
    
    for i in range(len(labels)):
        one_hot[i][labels[i]] = 1

    return one_hot
print (convert_to_one_hot(y))
