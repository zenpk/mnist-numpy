import gzip
import hashlib
import os

import numpy as np
import requests

# Fetch data (referenced code: https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb)
path = "./data"


def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# Validation split
rand = np.arange(60000)
np.random.shuffle(rand)
train_no = rand[:50000]
test_no = np.setdiff1d(rand, train_no)
X_train, X_test = X[train_no, :, :], X[test_no, :, :]
Y_train, Y_test = Y[train_no], Y[test_no]


# Initializing weights (-1 ~ 1 float np array)
def init(x, y):
    layer = np.random.uniform(-1., 1., size=(x, y)) / np.sqrt(x * y)
    return layer.astype(np.float32)


def d_sigmoid(x):
    return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)


# Softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def d_softmax(x):
    exp_element = np.exp(x - x.max())
    return exp_element / np.sum(exp_element, axis=0) * (1 - exp_element / np.sum(exp_element, axis=0))


# Forward and backward pass
def forward_backward_pass(x, y, w1, w2):
    targets = np.zeros((len(y), 10), np.float32)
    targets[range(targets.shape[0]), y] = 1

    # Forward

    x_w1 = x.dot(w1)
    x_relu = np.maximum(x_w1, 0)
    x_w2 = x_relu.dot(w2)
    out = softmax(x_w2)

    # Back Propagation

    error = 2 * (out - targets) / out.shape[0] * d_softmax(x_w2)
    update_w2 = x_relu.T @ error

    error = (w2.dot(error.T)).T * d_sigmoid(x_w1)
    update_w1 = x.T @ error

    return out, update_w1, update_w2


# Hyperparameters
hidden = 64
input_size = 28 * 28  # 784
seed = 666
batch = 32
epochs = 50000
test_interval = 1000
test_size = 10
lr = 0.0005
max_val = 255

np.random.seed(seed)
w1 = init(input_size, hidden)
w2 = init(hidden, 10)

losses, train_accuracy = [], []

for i in range(epochs):
    # randomize and create batches
    sample = np.random.randint(0, X_train.shape[0], size=batch)
    x = X_train[sample].reshape((-1, input_size)) / max_val
    y = Y_train[sample]

    out, update_w1, update_w2 = forward_backward_pass(x, y, w1, w2)
    train_out = np.argmax(out, axis=1)

    accuracy = (train_out == y).mean()
    train_accuracy.append(accuracy.item())

    loss = ((train_out - y) ** 2).mean()
    losses.append(loss.item())

    # SGD
    w1 = w1 - lr * update_w1
    w2 = w2 - lr * update_w2

    # testing our model using the test set every test_interval
    if (i + 1) % test_interval == 0:
        x = X_test.reshape((-1, input_size)) / max_val
        y = Y_test
        out, update_w1, update_w2 = forward_backward_pass(x, y, w1, w2)
        test_out = np.argmax(out, axis=1)
        test_accuracy = (test_out == y).mean()
        print(
            f"Epoch {i + 1}/{epochs}: train accuracy: {np.mean(train_accuracy):.3f}, test accuracy: {test_accuracy:.3f}")
        losses, train_accuracy = [], []

# Final accuracy
# x = X_test.reshape((-1, input_size))
# y = Y_test
# out, update_w1, update_w2 = forward_backward_pass(x, y, w1, w2)
# test_out = np.argmax(out, axis=1)
# test_accuracy = (test_out == y).mean()
# print(f"Final accuracy: {test_accuracy:.3f}")

# Random test
print(f"Random test:")
sample = np.random.randint(0, X_test.shape[0], size=test_size)
x = X_test[sample].reshape((-1, input_size)) / max_val
y = Y_test[sample]
out, update_w1, update_w2 = forward_backward_pass(x, y, w1, w2)
test_out = np.argmax(out, axis=1)
test_accuracy = (test_out == y).mean()
print(f"Testing label {y}, result: {test_out}, accuracy: {test_accuracy:.1f}")
