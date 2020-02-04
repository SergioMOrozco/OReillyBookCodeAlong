## Exercise 12 : 
# Implement Batch Gradient Descent with early stopping
# for Softmax Regression (without using Scitkit-Learn).

from sklearn import datasets
import soft_max_scratch
import numpy as np
iris = datasets.load_iris()

x = np.random.rand(1,2)
soft_clf = soft_max_scratch.SoftMaxClassifier(x, [[1,0,0]])
print(soft_clf.calculate_cross_entropy_gradient(0)) 



