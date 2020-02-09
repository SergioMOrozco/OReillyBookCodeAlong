## Exercise 12 : 
# Implement Batch Gradient Descent with early stopping
# for Softmax Regression (without using Scitkit-Learn).

from sklearn import datasets
import soft_max_scratch
import numpy as np


iris = datasets.load_iris()
data = iris["data"]
labels = iris["target"]

## Converts class labels to one hot encodings
def convert_to_one_hot(labels):

    class_count = len(set(labels)) 
    one_hot = np.zeros((len(labels),class_count))
    
    for i in range(len(labels)):
        one_hot[i][labels[i]] = 1

    return one_hot

labels_one_hot = convert_to_one_hot(labels)

## shuffle training data and labels (Gradient Descent performs better this way)
rand = np.random.permutation(len(data))
x,y = data[rand] , labels_one_hot[rand]

## Create and train SoftmaxClassifier on training data
soft_clf = soft_max_scratch.SoftMaxClassifier(learning_rate = 0.5, max_iter = 10000, minimum_step_size = 0.001, early_stopping = True)
soft_clf.train(x,y)


##TODO: Predict on test data. not testing data
y_pred = soft_clf.predict(x)

count = 0
for i in range (len (x)):
    if np.array_equal(y[i], y_pred[i]):
        count += 1
    
print (count / len(x))


