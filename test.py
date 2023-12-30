from cheras.Model.model import Model
from cheras.Layers.Dense import Dense
from cheras.Layers.Input import Input
from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

a = Input(784)
b = Dense(64, 'sigmoid', a)
c = Dense(64, 'sigmoid', b)
y = Dense(10, 'sigmoid', c)

model = Model (a, y)

X_train = np.array(X_train)
X_train = X_train.reshape (-1, 784)
X_train = X_train / 255

X_test = np.array(X_test)
X_test = X_test.reshape (-1, 784)
X_test = X_test / 255

model.optimize(0.001, 0.9, 0.999)
model.fit(X_train, Y_train, X_test, Y_test, 20, 32)