import numpy as np
import math

class SoftMaxClassifier ():

    def __init__(self,X,y):
        self._x = X
        self._y = y 
        self._class_count = len(self._y[0])
        self._weights = np.random.rand(self._class_count,X.shape[1])
        self._learning_rate = 0.1
        print (self._weights)

    def calculate_score(self,class_index,instance_index):
        weight = self._weights[class_index]
        return self._x[instance_index].dot(weight)

    def calculate_softmax(self,class_index,instance_index):
        sum_of_exp = 0
        for i in range (self._class_count):
            sum_of_exp += self.calculate_score(i,instance_index)

        return self.calculate_score(class_index,instance_index) / sum_of_exp

    def calculate_cross_entropy_gradient(self,class_index):
        sum = 0
        for i in range (len(self._x)):
            sum += ((self.calculate_softmax(class_index,i) - self._y[i][class_index]) * self._x[i])

        return sum / len(self._x)
    
    def calculate_new_weights(self,class_index):
        step_size = self.calculate_cross_entropy_gradient(class_index) * self._learning_rate
        return self._weights[class_index] - step_size




        


