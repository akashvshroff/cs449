import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, k, loss_fn):
        """
        Initialize the parameters for the model
        """
        self.W1 = np.random.rand(k, 2)
        self.b1 = np.zeros((k, 1))
        self.W2 = np.random.rand(1, k)
        self.b2 = np.zeros((1, 1))
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
        A1 = self.ReLU(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)

        return A1, A2

    def compute_loss(self, A2, Y):
        """
        Calculates loss based on given loss fn for all examples
        """
        n = Y.shape[1]  # num examples
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        loss = -np.sum(logprobs) / n
        return loss

    def back_prop(self, X, Y, A1, A2):
        """
        Calculates the partial derivatives for each of the parameters

        dsigmoid(z) = sigmoid(z)(1 - sigmoid(z))
        dBCE/dy' = -y/y' + (1-y)/(1-y')

        - X: Input data, shape (2, n)
        - Y: True labels, shape (1, n)
        - A1: Activations from hidden layer, shape (k, n)
        - A2: Output activations, shape (1, n)
        """
        n = X.shape[1]
        dZ2 = 0
        if self.loss_fn == "bce":  # binary cross entropy
            dZ2 = A2 - Y
        elif self.loss_fn == "mse":  # mean squared error
            pass
        else:
            raise "error: not implemented"
        dW2 = np.dot(dZ2, A1.T) / n
        db2 = np.sum(dZ2, axis=1, keepdims=True) / n
        dZ1 = np.dot(self.W2.T, dZ2)  # A1 * (1 - A1) sigmoid derivative
        dZ1[A1 <= 0] = 0  # relu derivative
        dW1 = np.dot(dZ1, X.T) / n
        db1 = np.sum(dZ1, axis=1, keepdims=True) / n

        return dW1, db1, dW2, db2

    def update_parameters(self, lr, dW1, db1, dW2, db2):
        """
        Returns the updated parameters
        """
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def sigmoid(self, z):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))

    def ReLU(self, z):
        """
        ReLU activation function
        """
        return np.maximum(0, z)

    def train(self, X, Y, lr, epochs):
        """
        Run the training loop and generate loss curves
        """
        loss_data = []
        for epoch in range(epochs):
            A1, A2 = self.forward_pass(X)
            loss = self.compute_loss(A2, Y)
            dW1, db1, dW2, db2 = self.back_prop(X, Y, A1, A2)
            self.update_parameters(lr, dW1, db1, dW2, db2)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, loss: {loss}")
                # print(f"dW1: {dW1}, db1: {db1}, dW2: {dW2}, db2: {db2}")
                # self.print_params()
                # input()

            loss_data.append(loss)
        return loss_data

    def predict(self, X):
        """
        Make predictions based on given X data and classify
        """

        _, A2 = self.forward_pass(X)

        Y = (A2 >= 0.5).astype(int)

        return Y

    def accuracy(self, Y, Y_pred):
        """
        Determine accuracy of predictions
        """
        return np.mean(Y == Y_pred)


def read_csv(filename):
    df = pd.read_csv(filename)
    X = df[["x1", "x2"]]
    Y = df["label"]

    X = X.to_numpy().T
    Y = Y.to_numpy().reshape(1, -1)

    return X, Y


def print_decision_boundary(model, X, Y, title):
    X_min, X_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    Y_min, Y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.01), np.arange(Y_min, Y_max, 0.01))

    # Predict classes for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    for label in np.unique(Y):
        plt.scatter(
            X[0, Y[0, :] == label],
            X[1, Y[0, :] == label],
            c=["red" if label == 1 else "blue"],
            label=f"Class {label}",
            marker="^" if label == 1 else "o",
            edgecolor="k",
        )  # '^' for triangles, 'o' for circles

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(f"plots/{title}")
    plt.show()


def xor_bce():
    k = 20
    XorBCE = MLP(k, "bce")
    X_train, Y_train = read_csv("xor_train.csv")
    XorBCE.train(X_train, Y_train, 0.1, 1000)

    # finetuning k values and num epochs using validation set
    X_valid, Y_valid = read_csv("xor_valid.csv")
    Y_pred = XorBCE.predict(X_valid)
    print(f"accuracy on validation set: {XorBCE.accuracy(Y_valid, Y_pred)}")

    # testing
    X_test, Y_test = read_csv("xor_test.csv")
    Y_pred = XorBCE.predict(X_test)
    print(f"accuracy on test set: {XorBCE.accuracy(Y_test, Y_pred)}")
    print_decision_boundary(XorBCE, X_test, Y_test, f"XOR BCE, k={k}")


def center_surround_bce():
    k = 16
    CSBCE = MLP(k, "bce")
    X_train, Y_train = read_csv("center_surround_train.csv")
    CSBCE.train(X_train, Y_train, 0.1, 800)

    # finetuning k values and num epochs using validation set
    X_valid, Y_valid = read_csv("center_surround_valid.csv")
    Y_pred = CSBCE.predict(X_valid)
    print(f"accuracy on validation set: {CSBCE.accuracy(Y_valid, Y_pred)}")

    # testing
    X_test, Y_test = read_csv("center_surround_test.csv")
    Y_pred = CSBCE.predict(X_test)
    print(f"accuracy on test set: {CSBCE.accuracy(Y_test, Y_pred)}")
    print_decision_boundary(CSBCE, X_test, Y_test, f"Center Surround BCE, k={k}")


def spiral_bce():
    k = 24
    SpiralBCE = MLP(k, "bce")
    X_train, Y_train = read_csv("spiral_train.csv")
    SpiralBCE.train(X_train, Y_train, 0.1, 1500)

    # finetuning k values and num epochs using validation set
    X_valid, Y_valid = read_csv("spiral_valid.csv")
    Y_pred = SpiralBCE.predict(X_valid)
    print(f"accuracy on validation set: {SpiralBCE.accuracy(Y_valid, Y_pred)}")

    # testing
    X_test, Y_test = read_csv("spiral_test.csv")
    Y_pred = SpiralBCE.predict(X_test)
    print(f"accuracy on test set: {SpiralBCE.accuracy(Y_test, Y_pred)}")
    print_decision_boundary(SpiralBCE, X_test, Y_test, f"Spiral BCE, k={k}")


def two_gaussians_bce():
    k = 20
    GaussianBCE = MLP(k, "bce")
    X_train, Y_train = read_csv("two_gaussians_train.csv")
    GaussianBCE.train(X_train, Y_train, 0.1, 1000)

    # finetuning k values and num epochs using validation set
    X_valid, Y_valid = read_csv("two_gaussians_valid.csv")
    Y_pred = GaussianBCE.predict(X_valid)
    print(f"accuracy on validation set: {GaussianBCE.accuracy(Y_valid, Y_pred)}")

    # testing
    X_test, Y_test = read_csv("two_gaussians_test.csv")
    Y_pred = GaussianBCE.predict(X_test)
    print(f"accuracy on test set: {GaussianBCE.accuracy(Y_test, Y_pred)}")
    print_decision_boundary(GaussianBCE, X_test, Y_test, f"Two Gaussians BCE, k={k}")


# xor_bce()
# center_surround_bce()
# spiral_bce()
# two_gaussians_bce()
