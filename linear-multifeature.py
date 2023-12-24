import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 

# for this example we will use the wine quality repo from UCI
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

# convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# when y has value > 7 it means the wine is good, otherwise it's bad
# so we will convert y to 0 and 1
y = (y > 7).astype(int)


print(X.shape, y.shape) 
# X has shape (N, 11) while y has shape (N,)
# N is the number of samples in the dataset

# split 80% of the data for training and 20% for testing
# we could also use sklearn.model_selection.train_test_split
# but we will do it manually here
N = X.shape[0]
train_size = int(0.8 * N)
X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# initialize the parameters
w = np.random.randn(X_train.shape[1])
b = np.random.randn()

# define the model
def model(x, w, b):
    return x @ w + b

# check if the model works
y_hat = model(X_train, w, b)

print(f"{w.shape=} {y_hat.shape=}")

# define the loss function
def loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# check if the loss function works
l = loss(y_train, y_hat)

# define the gradient function
def gradient(x, y, y_hat):
    y_hat = y_hat.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    dw = (2 / x.shape[0]) * x.T @ (y_hat - y)
    db = 2 * np.mean(y_hat - y)
    return dw, db
# check if the gradient function works
dw, db = gradient(X_train, y_train, y_hat)

# define the training loop
def train(x, y, epochs, lr):
    # initialize the parameters
    # w should consider the fact that x might be a batch of data
    # so we should initialize w with the shape of x
    w = np.random.randn(X_train.shape[1],1)
    b = np.random.randn()
    # loop over the epochs
    for epoch in range(epochs):
        # forward pass
        y_hat = model(x, w, b)
        # compute the loss
        l = loss(y, y_hat)
        # compute the gradients
        dw, db = gradient(x, y, y_hat)
        # update the parameters
        w -= lr * dw
        b -= lr * db
        # print the loss every 100 epochs
        if epoch % 5000 == 0:
            print(f"Epoch {epoch} loss {l}")
    return w, b

# train the model
w, b = train(X_train, y_train, 50000, 5e-5)

# also calculate the accuracy on the test set
y_hat = model(X_test, w, b)
l = loss(y_test, y_hat)
print(f"Test loss {l}")
y_hat = np.round(y_hat)
print(f"{y_hat[0]=} {y_test[0]=}")

# consider the fact that y is a value between 3 and 9 when calculating the accuracy
# so we need to round y_hat to the nearest integer

# calculate the accuracy
accuracy = np.mean(y_hat == y_test)
print(f"Accuracy {accuracy}")

# plot the data
plt.scatter(X_test[:, 0], y_test, label="true")
plt.scatter(X_test[:, 0], y_hat, label="predicted")
plt.legend()
plt.show()

