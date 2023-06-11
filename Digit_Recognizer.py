import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("D:\Programming Files\pyt\.vscode\MNIST data\mnist_train1.csv")

data = np.array(data)
np.random.shuffle(data)

m, n = data.shape # n is number of pics, m is the number of pixels


data_test = data[0:20000].T # now its flipped
Y_test = data_test[0]
X_test = data_test[1:m]
X_test = X_test/255


data_train = data[20000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255

####################  Code  #########################

def essential_param():
    w1 = np.random.rand(16, 784) - 0.5
    b1 = np.random.rand(16, 1) - 0.5
    w2 = np.random.rand(16, 16) - 0.5
    b2 = np.random.rand(16, 1) - 0.5
    w3 = np.random.rand(10, 16) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2, w3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def diff_ReLU(Z):
    return Z > 0

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def forward_prop(w1, b1, w2, b2, w3, b3, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)

    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)

    z3 = w3.dot(a2) + b3
    a3 = ReLU(z3)
    return z1, a1, z2, a2, z3, a3 

def right_label(Y):
    right_Y = np.zeros((Y.size, Y.max() + 1))
    right_Y[np.arange(Y.size), Y] = 1
    right_Y = right_Y.T
    return right_Y

def back_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, x, y):
    rightY = right_label(y)
    dz3 = a3 - rightY
    dw3 = 1/m * dz3.dot(a2.T)
    db3 = 1/m * np.sum(dz3)

    dz2 = w3.T.dot(dz3) * diff_ReLU(z2)
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m * np.sum(dz2)

    dz1 = w2.T.dot(dz2) * diff_ReLU(z1)
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1)
    return dw1, db1, dw2, db2, dw3, db3

def update_param(dw1, w1, db1, b1, dw2, w2, db2, b2, dw3, w3, db3, b3, alpha):
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    w3 = w3 - alpha * dw3

    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    b3 = b3 - alpha * db3
    return w1, b1, w2, b2, w3, b3

def get_predictions(a3):
    return np.argmax(a3, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def gradiant_descent(x, y, alpha, iterations, period): 
    w1, b1, w2, b2, w3, b3 = essential_param()
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, x)
        
        dw1, db1, dw2, db2, dw3, db3 = back_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, x, y)

        w1, b1, w2, b2, w3, b3 = update_param(dw1, w1, db1, b1, dw2, w2, db2, b2, dw3, w3, db3, b3, alpha)

        if i % period == 0:
            print("iteration ", i)
            predections = get_predictions(a3)
            print(get_accuracy(predections, y))
    return w1, b1, w2, b2, w3, b3

def test_out(index, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X_test[:, index, None])

    predicton = get_predictions(a3)
    label = Y_test[index]
    print("predection: ", predicton)
    print("Label: ", label)


def randomInteger():
    unfloor = 20000 * np.random.rand()
    return int(np.floor(unfloor))


def write_data(data, file):
    arr = []
    f = open(file, "w")

    m, n = data.shape
    data = data.astype(np.str_)

    for i in range(m):
        for j in range(n):
            arr.append(data[i, j])

    strarr = ",".join(arr)

    f.write(strarr)
    f.close()


w1, b1, w2, b2, w3, b3 = gradiant_descent(X_train, Y_train, 0.1, 100, 10)



#############  Testing  Data
#test_out(randomInteger(), w1, b1, w2, b2, w3, b3)
#
#test_out(randomInteger(), w1, b1, w2, b2, w3, b3)
#
#test_out(randomInteger(), w1, b1, w2, b2, w3, b3)
#
#test_out(randomInteger(), w1, b1, w2, b2, w3, b3)
#
#test_out(randomInteger(), w1, b1, w2, b2, w3, b3)
#
#test_out(randomInteger(), w1, b1, w2, b2, w3, b3)


########### Registering Numbers

#write_data(w1, "w1.txt")
#write_data(w2, "w2.txt")
#write_data(w3, "w3.txt")
#
#write_data(b1, "b1.txt")
#write_data(b2, "b2.txt")
#write_data(b3, "b3.txt")





