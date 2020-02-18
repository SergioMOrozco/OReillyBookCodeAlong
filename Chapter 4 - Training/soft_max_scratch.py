import numpy as np
import math
from sklearn.metrics import accuracy_score 

class SoftMaxClassifier ():

    def __init__(self,
                learning_rate = 0.1,
                max_iter = 1000,
                minimum_step_size = 0.001,
                early_stopping = True,
                n = 20):
        """
        Softmax Classifer using Batch Gradient Descent and Early Stopping.

        Keyword Arguments:
        learning_rate -- the amount that the weights for Gradient Descent are updated during training. 
        max_iter -- the max number of iterations that Gradient Descent will perform.
        minimum_step_size -- the minimum step size that will be allowed during training. Stops training if minimum is reached.
        early_stopping -- flag to determine whether or not to stop training if validation error begins to increase.
        n - Only used for early_stopping. The max number of times that the validation error is allowed to go up sequentially.
        """

        self.__learning_rate = learning_rate
        self.__max_iter = max_iter
        self.__minimum_step_size = minimum_step_size
        self.__early_stopping = early_stopping
        self.__n = n

    """ 
    Calculates the weighted sum of the input
    features for a given instance x, for a
    given class k.
    """
    def __calculate_score(self,k,x):
        weight = self.__weights[k]
        return x.dot(weight)

    """
    Calculates the probability that a given
    instance x belongs to a given class k
    """
    def __calculate_softmax(self,k,x):
        sum_of_exp = 0
        for i in range (self.__class_count):
            sum_of_exp += self.__calculate_score(i,x)

        return self.__calculate_score(k,x) / sum_of_exp

    """
    Calculates the cross entropy cost function
    for every instance in X, for a given class k.
    """
    def __calculate_cross_entropy_gradient(self,k):
        sum = 0
        for i in range (len(self.__x)):
            sum += ((self.__calculate_softmax(k,self.__x[i]) - self.__y[i][k]) * self.__x[i])

        return sum / len(self.__x)
    """
    subtracts the old weights from the step size
    and returns them, for a given class k.
    """
    def __calculate_new_weights(self,k):

        ## dont update if the class index is in the exclusion list
        if k in self.__exclusion_list:
            return self.__weights[k]

        step_size = self.__calculate_cross_entropy_gradient(k) * self.__learning_rate
        
        """
        add class index to exclusion list if all
        the step sizes are below the minimum
        """
        if np.amax(step_size) <= self.__minimum_step_size:
            self.__exclusion_list.append(k)

        return self.__weights[k] - step_size

    def train(self,X,y):
        """
        trains the SoftMaxClassifier on the provided data

        Keyword Arguments:
        X -- the training data : (m,n) numpy array
        y -- the target data (labels) : (m,k) one hot encoded numpy array
        """
        if (self.__early_stopping):
            split_index = int(len(X) * 0.1)
            self.__x = X[split_index:]
            self.__y = y[split_index:]
            self.__x_val = X[:split_index]
            self.__y_val = y[:split_index]
        else:
            self.__x = X
            self.__y = y 
            
        self.__class_count = len(self.__y[0])
        self.__weights = np.random.rand(self.__class_count,X.shape[1])
        self.__exclusion_list = []
        
        max_acc = -1 

        n = 0
        for i in range(self.__max_iter):
            for j in range(self.__class_count):
                self.__weights[j] = self.__calculate_new_weights(j)
            
            """ 
            if all the step sizes for
            each weight reach the minimum,
            stop training
            """
            if len(self.__exclusion_list) == self.__class_count:
                break
            
            """
            if early stopping is enabled, we stop
            validation error from going up due to overfitting
            """
            if (self.__early_stopping):
                y_val_pred = self.predict(self.__x_val)
                acc = accuracy_score(self.__y_val,y_val_pred)
                if acc > max_acc:
                    max_acc = acc 
                    n = 0
                else:
                    n += 1
                    if n >= self.__n:
                        break

    def predict(self,X):
        """
        predicts the labels for the given data

        Keyword Arguments:
        X -- data to predict labels

        Returns:
        a one hot encoded numpy array
        """
        y = np.zeros((len(X),self.__class_count))
        for i in range(len(X)):
            max_score_index = 0
            max_score = 0
            for j in range (self.__class_count):
                score = self.__calculate_softmax(j,X[i])
                if score > max_score:
                    max_score = score

                    max_score_index = j

            y[i][max_score_index] = 1

        return y
