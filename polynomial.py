import numpy as np
import matplotlib.pyplot as plt


# this function will map R2 to R
def f(x1, x2):
    # adds noise
    return x1 ** 2 + x2 ** 2 + np.random.randn(len(x1))


# generate some data
stop = 10

x1 = np.linspace(-stop, stop, 100)
x2 = np.linspace(-stop, stop, 100)
y = f(x1, x2)

x = np.array([[x1[i], x2[i]] for i in range(len(x1))])

print(x[:5], y[:5])

# # we can plot the data in 3D
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x1, x2, y)
# plt.show()


# initialize the parameters
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
print(f"{w1=} {w2=} {b=}")

# define the model knowing it's a polynomial of degree 2
def model(x, w1, w2, b):
    return w1 * x[:, 0]**2 + w2 * x[:, 1]**2 + b

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
    dw1 = np.mean(2 * x[:, 0]**2 * (y_hat - y))
    dw2 = np.mean(2 * x[:, 1]**2 * (y_hat - y))
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
    return dw1, dw2, db

# define the training loop
def train(x, y, epochs, lr):
    # initialize the parameters
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()

    # loop over the epochs
    for epoch in range(epochs):
        # compute the output of the model
        y_hat = model(x, w1, w2, b)
        # compute the loss
        l = loss(y, y_hat)
        # compute the gradients
        dw1, dw2, db = gradient(x, y, y_hat)
        # update the parameters
        w1 -= lr * dw1
        w2 -= lr * dw2
        b -= lr * db
        # print the loss
        print(f"Epoch {epoch+1}/{epochs}, loss={l:.3f}")
    return w1, w2, b

# train the model
w1, w2, b = train(x, y, 10, 0.0001)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y)
# plot the model
x1, x2 = np.meshgrid(x1, x2)
y_hat = w1 * x1**2 + w2 * x2**2 + b
ax.plot_surface(x1, x2, y_hat, alpha=0.5)
plt.show()
