import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)  
# here we use a common regression model w*x + b, combined with a sigmoid
# activation function to get a classification model that can classify
# linearly separable data (data that can be separated by a line)

# we will use the iris dataset, which contains 3 classes of flowers
# we will only use 2 classes, so we can plot the data in 3D
# the data is linearly separable, so we can use a linear model to classify it

# load the data from the csv file Iris.csv
data = np.genfromtxt("Iris.csv", delimiter=",", skip_header=1)

# the data contains these columns: Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
# we will only use the first 2 columns and the last column
# the first 2 columns are the features, and the last column is the label

# the last column can be Iris-setosa, Iris-versicolor or Iris-virginica
# we limit the data to only 2 classes, so we will only use Iris-setosa and Iris-versicolor

# we can map Iris-setosa to 0 and Iris-versicolor to 1
# we can use a sigmoid activation function to get a value between 0 and 1
# if the value is greater than 0.5, we can classify the data as Iris-versicolor
# if the value is less than 0.5, we can classify the data as Iris-setosa
# only take data where last column is Iris-setosa or Iris-versicolor
data = data[data[:, 5] != "Iris-virginica"]
# map Iris-setosa to 0 and Iris-versicolor to 1
data[:, 5] = np.where(data[:, 5] == "Iris-setosa", 0, 1)

# split training and testing data using the ratio 80:20

# shuffle the data
np.random.shuffle(data)
# split the data
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

x1 = train_data[:, 1] # SepalLengthCm
x2 = train_data[:, 2] # SepalWidthCm
y = train_data[:, 5]



x = np.array([[x1[i], x2[i]] for i in range(len(x1))])

# we can plot the data in 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y)
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# initialize the parameters
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
print(f"{w1=} {w2=} {b=}")

# define the model
def model(x, w1, w2, b):
    return sigmoid(w1 * x[:, 0] + w2 * x[:, 1] + b)

# define the loss function
def loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# define the gradient function
def gradient(x, y, y_hat):
    # the loss function has the form of (y - y_hat) ** 2
    # using the chain rule we can get the gradient of the loss function
    # with respect to w and b
    # w appears in the loss function in the form of y_hat = sigmoid(w * x + b)
    # so if we write the loss function as (y - sigmoid(w * x + b)) ** 2
    # we can use the chain rule to get the gradient of the loss function
    # with respect to w
    # dL/dw = dL/dy_hat * dy_hat/dw
    # dy_hat/dw = sigmoid_derivative(w * x + b) * x
    # dL/dw = dL/dy_hat * sigmoid_derivative(w * x + b) * x
    # dL/dy_hat = 2 * (y - y_hat)
    # dL/dw = 2 * (y - y_hat) * sigmoid_derivative(w * x + b) * x
    dw1 = np.mean(2 * (y_hat - y) * sigmoid_derivative(y_hat) * x[:, 0])
    dw2 = np.mean(2 * (y_hat - y) * sigmoid_derivative(y_hat) * x[:, 1])
    # b appears in the loss function in the form of y_hat = sigmoid(w * x + b)
    # so if we write the loss function as (y - sigmoid(w * x + b)) ** 2
    # we can use the chain rule to get the gradient of the loss function
    # with respect to b
    # dL/db = dL/dy_hat * dy_hat/db
    # dy_hat/db = sigmoid_derivative(w * x + b)
    # dL/db = dL/dy_hat * sigmoid_derivative(w * x + b)
    # dL/dy_hat = 2 * (y - y_hat)
    # dL/db = 2 * (y - y_hat) * sigmoid_derivative(w * x + b)
    db = np.mean(2 * (y_hat - y) * sigmoid_derivative(y_hat))

    return dw1, dw2, db

# define the training loop
def train(x, y, epochs, lr):
    # initialize the parameters
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()

    
    # loop over the epochs
    for epoch in range(epochs):
        # forward pass
        y_hat = model(x, w1, w2, b)
        # compute the loss
        l = loss(y, y_hat)
        # compute the gradients
        dw1, dw2, db = gradient(x, y, y_hat)
        # update the parameters
        w1 -= lr * dw1
        w2 -= lr * dw2
        b -= lr * db
    return w1, w2, b

# train the model
w1, w2, b = train(x, y, 1000, 0.0001)

# evaluate the model and its accuracy
x1 = test_data[:, 1] # SepalLengthCm
x2 = test_data[:, 2] # SepalWidthCm

x = np.array([[x1[i], x2[i]] for i in range(len(x1))])

y = test_data[:, 5]

y_hat = model(x, w1, w2, b)

# round the predictions to 0 or 1
y_hat = np.round(y_hat)

# calculate the accuracy
accuracy = np.mean(y_hat == y)
print(f"accuracy: {accuracy}")

# plot the data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y)
ax.scatter(x1, x2, y_hat)

# draw the hyperplane that separates the data
# the hyperplane is w1 * x1 + w2 * x2 + b = 0
# we can rewrite it as x2 = -(w1 * x1 + b) / w2
# we can plot the hyperplane in 3D
# we can plot the hyperplane in 2D

# plot the hyperplane in 3D
x1 = np.linspace(4, 7, 100)
x2 = np.linspace(2, 4.5, 100)
x1, x2 = np.meshgrid(x1, x2)
x3 = -(w1 * x1 + w2 * x2 + b) / w2
ax.plot_surface(x1, x2, x3, alpha=0.5)

plt.show()

