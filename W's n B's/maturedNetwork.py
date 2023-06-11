import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#########  DATA

data = pd.read_csv("mnist_train1.csv")

data = np.array(data)
np.random.shuffle(data)

m, n = data.shape # n is number of pics, m is the number of pixels


data_test = data[0:20000].T # now its flipped
Y_test = data_test[0]
X_test = data_test[1:m]
X_test = X_test/255


############ Initializing Functions
def readWnB(path, rows, col):
    WnB = open(path, "r")
    return np.array(WnB.read().split(",")).reshape((rows, col)).astype(float)

def ReLU(Z):
    return np.maximum(Z, 0)

def forward_prop(w1, b1, w2, b2, w3, b3, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)

    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)

    z3 = w3.dot(a2) + b3
    a3 = ReLU(z3)
    return a3 

def get_predictions(a3):
    return np.argmax(a3, 0)

def test_Label(index, w1, b1, w2, b2, w3, b3):
    a3 = forward_prop(w1, b1, w2, b2, w3, b3, X_test[:, index, None])

    predicton = get_predictions(a3)
    label = Y_test[index]
    print("predection: ", predicton)
    print("Label: ", label)


def test_pic(index, w1, b1, w2, b2, w3, b3):
    a3 = forward_prop(w1, b1, w2, b2, w3, b3, X_test[:, index, None])
    predicion = get_predictions(a3)
    print("Prediction, ", predicion)


    curren_img = X_test[:, index, None].reshape((28,28)) * 255
    plt.gray()
    plt.imshow(curren_img, interpolation="nearest")
    plt.show()
    
def randomInteger():
    unfloor = 20000 * np.random.rand()
    return int(np.floor(unfloor))
    




############ CODE ###############

w1 = readWnB("D:\Programming Files\pyt\.vscode\MNIST Neural Network\W's n B's\w1.txt", 16, 784)
b1 = readWnB("D:\Programming Files\pyt\.vscode\MNIST Neural Network\W's n B's\p1.txt", 16, 1)

w2 = readWnB("D:\Programming Files\pyt\.vscode\MNIST Neural Network\W's n B's\w2.txt", 16, 16)
b2 = readWnB("D:\Programming Files\pyt\.vscode\MNIST Neural Network\W's n B's\p2.txt", 16, 1)

w3 = readWnB("D:\Programming Files\pyt\.vscode\MNIST Neural Network\W's n B's\w3.txt", 10, 16)
b3 = readWnB("D:\Programming Files\pyt\.vscode\MNIST Neural Network\W's n B's\p3.txt", 10, 1)



test_pic(randomInteger(), w1, b1, w2, b2, w3, b3)