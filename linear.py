# this script will contain a basic linear regression model with only numpy

import numpy as np
import matplotlib.pyplot as plt

# generate some data
stop = 10

x = np.linspace(0, stop, 100)
y = 2 * (x) + 100 + np.random.randn(100)

# plot the data
# plt.scatter(x, y)
# plt.show()


# initialize the parameters
w = 0 #np.random.randn()
b = 0 #np.random.randn()
print(f"{w=} {b=}")

# define the model
def model(x, w, b):
    print(f"{x=} {w=} {b=}")
    return w * x + b

# define the loss function
def loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# define the gradient function
def gradient(x, y, y_hat):
    # the loss function has the form of (y - y_hat) ** 2
    # using the chain rule we can get the gradient of the loss function
    # with respect to w and b
    # w appears in the loss function in the form of y_hat = w * x + b
    # so if we write the loss function as (y - (w * x + b)) ** 2
    # we can use the chain rule to get the gradient of the loss function
    # with respect to w
    # dL/dw = dL/dy_hat * dy_hat/dw
    # dy_hat/dw = x
    # dL/dw = dL/dy_hat * x
    # dL/dy_hat = 2 * (y - y_hat)
    # dL/dw = 2 * (y - y_hat) * x
    dw = np.mean(2 * x * (y_hat - y))
    # b appears in the loss function in the form of y_hat = w * x + b
    # so if we write the loss function as (y - (w * x + b)) ** 2
    # we can use the chain rule to get the gradient of the loss function
    # with respect to b
    # dL/db = dL/dy_hat * dy_hat/db
    # dy_hat/db = 1
    # dL/db = dL/dy_hat * 1
    # dL/dy_hat = 2 * (y - y_hat)
    # dL/db = 2 * (y - y_hat)
    db = np.mean(2 * (y_hat - y))
    return dw, db

# define the training loop
def train(x, y, epochs, lr):
    # initialize the parameters
    # w should consider the fact that x might be a batch of data
    # so we should initialize w with the shape of x
    w = np.zeros_like(x)
    b = 0

    
    # loop over the epochs
    for epoch in range(epochs):
        # forward pass
        y_hat = model(x, w, b)
        # compute the loss
        l = loss(y, y_hat)
        # compute the gradients
        dw, db = gradient(x, y, y_hat)
        print(f"{dw=} {db=} {w=} {b=}")
        # update the parameters, this is gradient descent, so we
        # subtract the gradient multiplied by the learning rate or add?
        # well, we want to minimize the loss, so we subtract
        w -= lr * dw
        b -= lr * db
        # print the loss
        print(f"epoch: {epoch} loss: {l} w: {w} b: {b}")
        # break
    return w, b

# train the model
w, b = train(x, y, 1000, 0.02)

# plot the data
plt.scatter(x, y)

# plot the predictions
plt.plot(x, model(x, w, b), color="red")

# show the plot
plt.show()

# print the parameters
print(f"w: {w} b: {b}")