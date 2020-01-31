## Exercise 12 : 
# Implement Batch Gradient Descent with early stopping
# for Softmax Regression (without using Scitkit-Learn).

from sklearn import datasets
import soft_max_scratch
import numpy as np
iris = datasets.load_iris()

soft_clf = soft_max_scratch.SoftMaxClassifier(np.random.rand(1,2), [0,1,2])
print(soft_clf.calculate_softmax(0))



