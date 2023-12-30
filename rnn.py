import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) 
input_size = 1
seq_length = 20
hidden_size = 10
output_size = 1
num_layers = 1

# generate some data
N = 1000
train_size = int(0.7 * N)
test_size = N - train_size

x = np.sin(np.linspace(0, 10 * np.pi, N)) + np.random.randn(N) * 0.1

# plot the data
# plt.plot(x)
# plt.show()

# we want to predict the next value of sin(x) based on the previous 10 values
# of sin(x)

# split the data into sequences
X = []
y = []
for i in range(N - seq_length):
    X.append(x[i:i + seq_length])
    y.append(x[i + seq_length])

# convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

sample = 0


# split the data into training and testing
X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

class RNN():
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # initialize the parameters
        self.Wx = np.random.randn(input_size, hidden_size)
        self.Wh = np.random.randn(hidden_size, hidden_size)
        self.Wy = np.random.randn(hidden_size, output_size)
        self.b = np.random.randn(hidden_size)

        self.hidden_states = []
        self.y_hats = []
    
    def forward(self, X):
        # initialize the hidden state
        h = np.zeros((self.hidden_size, ))
        # initialize the output
        y = np.zeros((X.shape[0], self.output_size))
        # loop through the sequence

        self.hidden_states = []
        self.y_hats = []
        for t in range(X.shape[0]):
            # get the current input
            x = X[t]
            # compute the current hidden state
            h = np.tanh(np.dot(x, self.Wx) + np.dot(h, self.Wh) + self.b)
            # compute the current output
            y[t] = np.dot(h, self.Wy)
            # save the hidden state
            self.hidden_states.append(h)
            self.y_hats.append(y[t])
        return y
    
    def backward(self, X, y, lr):  
        dWx = 0
        dWy = 0
        dWh = 0
        db = 0
        dL_dy_hat = 0
        # loop through the sequence backwards
        for t in reversed(range(X.shape[0])):
            # print t
            # get the current input
            x = X[t]
            # get the current hidden state
            h = self.hidden_states[t]
            # get the current output
            y_hat = self.y_hats[t]
            # compute the gradient of the loss function with respect to y_hat
            dL_dy_hat = 2 * (y_hat - y)
            # compute the gradient of the loss function with respect to Wy
            dWy += np.dot(h.reshape(-1, 1), dL_dy_hat.reshape(1, -1))
            # compute the gradient of the loss function with respect to by
            db += dL_dy_hat
            # compute the gradient of the loss function with respect to h
            dL_dh = np.dot(dL_dy_hat, self.Wy.T)
            # compute the gradient of the loss function with respect to Wh
            dWh += np.dot(self.hidden_states[t - 1].reshape(-1, 1), dL_dh.reshape(1, -1))
            # compute the gradient of the loss function with respect to Wx
            dWx += np.dot(x.reshape(-1, 1), dL_dh.reshape(1, -1))
        
        self.Wx -= lr * dWx
        self.Wy -= lr * dWy
        self.Wh -= lr * dWh
        self.b -= lr * db


    
    
    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # get the current input
                X_i = X[i]
                y_i = y[i]
                # compute the current output
                y_hat = self.forward(X_i)
                self.backward(X_i, y_i, lr)
                if epoch % 100 == 0 and i == seq_length - 1:
                    print(f"{np.mean((y_hat - y_i) ** 2)=}")

# define the model
model = RNN(input_size, hidden_size, num_layers, output_size)
model.train(X_train, y_train, 200, 0.00001)

print(f"{X_test.shape[0]=}")
y_hats_test = []
# test the model
for i in range(X_test.shape[0]):
    X_i = X_test[i]
    y_i = y_test[i]
    y_hat = model.forward(X_i)
    y_hats_test.append(y_hat[-1])

y_hats_test = np.array(y_hats_test)
#plot the predictions
plt.plot(y_test, label="y_test")
plt.plot(y_hats_test, label="y_hats_test")

plt.legend()


plt.show()