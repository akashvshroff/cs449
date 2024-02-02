import numpy as np
import pandas as pd

np.random.seed(42)


class MLP:
    def __init__(self, k, lr, loss_fn):
        """
        Initialize the parameters for the model
        """
        self.W1 = np.random.rand(k, 2)
        self.b1 = np.random.rand(k, 1)
        self.W2 = np.random.rand(1, k)
        self.b2 = np.random.rand(1, 1)
        self.lr = lr
        self.loss_fn = loss_fn
        pass

    def print_params(self):
        print("W1", self.W1)
        print("b1", self.b1)
        print("W2", self.W2)
        print("b2", self.b2)

    def forward_pass(self, X):
        """
        Given a set of training examples, return the predicted values for all examples
        """
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.sigmoid(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)

        return A1, A2

    def compute_loss(A2, Y):
        """
        Calculates loss based on given loss fn for all examples
        """
        n = Y.shape[1]  # num examples
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        loss = -np.sum(logprobs) / n
        return loss

    def back_prop(self):
        """
        Calculates the partial derivatives for each of the parameters
        """
        pass

    def update_parameters(self):
        """
        Returns the updated parameters
        """
        pass

    def sigmoid(self, z):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))


mlp = MLP(3, 0, None)
mlp.print_params()
X = np.random.rand(3, 2)
X = X.T
print(mlp.forward_pass(X))
