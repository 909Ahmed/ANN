from model import Model
from layers import Layer
from Input import Input
from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, Y_train), (X_test, y_test) = mnist.load_data()

a = Input(784)
b = Layer(30, 'sigmoid', a)
c = Layer(30, 'sigmoid', b)
y = Layer(10, 'sigmoid', c)

model = Model (a, y)

X_train = np.array(X_train)
X_train = X_train.reshape (-1, 784)

X_train = X_train / 255
model.fit(X_train[:10000], Y_train[:10000], 10, 32)