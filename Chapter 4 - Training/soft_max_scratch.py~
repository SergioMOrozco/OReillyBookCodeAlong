import numpy as np
import math

class SoftMaxClassifier ():

    def __init__(self, learning_rate = 0.1 , max_iter = 100):
        self._learning_rate = learning_rate
        self._max_iter = max_iter

    def __calculate_score(self,class_index,x):
        weight = self._weights[class_index]
        return x.dot(weight)

    def __calculate_softmax(self,class_index,x):
        sum_of_exp = 0
        for i in range (self._class_count):
            sum_of_exp += self.__calculate_score(i,x)

        return self.__calculate_score(class_index,x) / sum_of_exp

    def __calculate_cross_entropy_gradient(self,class_index):
        sum = 0
        for i in range (len(self._x)):
            sum += ((self.__calculate_softmax(class_index,self._x[i]) - self._y[i][class_index]) * self._x[i])

        return sum / len(self._x)
    
    def __calculate_new_weights(self,class_index):
        step_size = self.__calculate_cross_entropy_gradient(class_index) * self._learning_rate
        return self._weights[class_index] - step_size

    def train(self,X,y):
        self._x = X
        self._y = y 
        self._class_count = len(self._y[0])
        self._weights = np.random.rand(self._class_count,X.shape[1])

        for i in range(self._max_iter):
            for j in range(self._class_count):
                self._weights[j] = self.__calculate_new_weights(j)

    def predict(self,X):
        y = np.zeros((len(X),self._class_count))
        for i in range(len(X)):
            max_score_index = 0
            max_score = 0
            for j in range (self._class_count):
                score = self.__calculate_softmax(j,X[i])
                if score > max_score:
                    max_score = score
                    max_score_index = j

            y[i][max_score_index] = 1

        return y

            



        


