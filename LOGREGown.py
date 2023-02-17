#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        # initialize parameters
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        # gradient descent
        for i in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(z)
            dw = (1/m) * np.dot(X.T, (y_hat - y))
            db = (1/m) * np.sum(y_hat - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(z)
        return np.round(y_hat)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# create example data
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12]])
y = np.array([0, 0, 0, 1, 1, 1])

# create logistic regression object
lr = LogisticRegression()

# fit the model to the data
lr.fit(X, y)

# make predictions on new data
X_new = np.array([[2, 5], [4, 9],[3,6],[1,2]])
y_pred = lr.predict(X_new)

print("Predictions:", y_pred)


# In[5]:


from sklearn.linear_model import LogisticRegression
import numpy as np

# create example data
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12]])
y = np.array([0, 0, 0, 1, 1, 1])

# create logistic regression object
lr = LogisticRegression()

# fit the model to the data
lr.fit(X, y)

# make predictions on new data
X_new = np.array([[2, 5], [4, 9]])
y_pred = lr.predict(X_new)

print("Predictions:", y_pred)


# In[ ]:




