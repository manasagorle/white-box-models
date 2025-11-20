import numpy as np

class Model:
    def __init__(self, learning_rate=0.05, max_iterations=100):
        self.__learning_rate = learning_rate
        self.__max_iterations = max_iterations
        self.__weights = np.zeros(1)
        self.__constant = 0
    
    def __str__(self):
        return "Linear Regression Model"

    def fit(self, X, y):
        self.__weights = np.array([0.0 for i in range(X.shape[1])])
        self.__compute_weights_and_constant(X, y)
        self.__residual_squared_sum = self.__residual_squared_sum(X, y)
        self.print_results()

    def __compute_weights_and_constant(self, X, y):
        for i in range(self.__max_iterations):
            y_predicted = self.predict(X)
            gradient_weights = (1/len(y)) * X.T.dot(y - y_predicted)
            gradient_constant = (1/len(y)) * np.sum(y - y_predicted)
            self.__weights -= self.__learning_rate*gradient_weights
            self.__constant -= self.__learning_rate*gradient_constant

    def predict(self, X):
        return X.dot(self.__weights) + self.__constant
    
    def __residual_squared_sum(self, X, y):
        return np.sum(np.square(X.dot(self.__weights) - y))
    
    def print_results(self):
        print("Residual Squared Sum:", self.__residual_squared_sum)
        print("Coefficients:", self.__weights)
        print("Intercept:", self.__constant)

if __name__ == '__main__':
    lm = Model()
    lm.fit(np.array([[1, 2], [2, 4]]), np.array([3, 6]))
    print(lm)