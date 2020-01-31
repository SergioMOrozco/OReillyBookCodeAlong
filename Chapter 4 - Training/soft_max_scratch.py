import numpy as np
import math

class SoftMaxClassifier ():

    def __init__(self,X,y):
        self._x = X
        self._y = y 
        self._class_count = len(np.unique(y))
        self._weights = np.random.rand(self._class_count,X.shape[1])

    def calculate_score(self,class_index):
        weight = self._weights[class_index]
        return self._x.dot(weight)

    def calculate_softmax(self,class_index):
        sum_of_exp = 0
        for i in range (self._class_count):
            sum_of_exp += self.calculate_score(i)

        return self.calculate_score(class_index) / sum_of_exp

    def calculate_cross_entropy(self,class_index):
        pass


        


